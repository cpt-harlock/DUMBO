use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use crate::packet::{Packet, PacketReceiver, PacketSource};
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::cell::RefCell;
use std::iter::Iterator;
use crate::flow_manager::{FAKE_SYN, FlowManager, prepare_aggregated_data};


pub struct HierarchicalFlowManager {
    layers: Vec<HashMap<usize, Vec<(u128,(f64, Vec<Packet>), usize)>>>,
    // one value for each layer, both for the slots
    pub rows: Vec<usize>,
    slots:  Vec<usize>,
    seed: usize,
    evicted_receiver: Option<Rc<RefCell<dyn PacketReceiver>>>,
    model_receiver: Option<Rc<RefCell<dyn PacketReceiver>>>,
    evicted_cache_stats_vector: Vec<(f64,String,usize,usize,usize)>,
    collected_cache_stats_vector: Vec<(f64,String,usize,usize,usize)>,
    // it also determines the number of layer, which is packet_limit - 1
    packet_limit: usize,
    fake_syn_set: HashSet<u128>,
}

impl HierarchicalFlowManager {
    pub fn build(packet_limit: usize, proportions: Vec<f64>, seed: usize, slots: usize, total_entries: usize) -> HierarchicalFlowManager {
        // assert packet_limit is coherent with poportions vector
        assert!(proportions.len() == (packet_limit - 1));
        // assert sum of proporions is 1.0
        //assert!(proportions.iter().fold(0.0, |acc,e| { acc + *e}) == 1.0);
        let mut rows_vec: Vec<usize> = vec![];
        let mut layers: Vec<HashMap<_,_>> = vec![];
        for i in proportions {
            rows_vec.push(((total_entries as f64 *i)) as usize);
            layers.push(HashMap::new());
        }
        HierarchicalFlowManager {
            layers,
            rows: rows_vec,
            slots: vec![slots; packet_limit-1],
            seed,
            evicted_receiver: None,
            model_receiver: None,
            evicted_cache_stats_vector: vec![],
            collected_cache_stats_vector: vec![],
            packet_limit,
            fake_syn_set: HashSet::new(),
        }
    }

    fn update_new_flow_counter(&mut self, new_key: u128) {
        for layer in &mut self.layers {
            for bucket in layer {
                for slot in bucket.1 {
                    if slot.0 != new_key {
                        slot.2 += 1;
                    }
                }
            }
        }
    }

    fn insert_into_layer(&mut self, value: (u128, (f64, Vec<Packet>), usize), layer_index: usize, timestamp: f64) -> Option<(u128, (f64, Vec<Packet>), usize)> {
        assert!(layer_index < (self.packet_limit - 1));
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        layer_index.hash(&mut hasher);
        let key = value.0;
        key.hash(&mut hasher);
        let index = hasher.finish() as usize % self.rows[layer_index] ;
        let mut ret = None;

        if let Some(bucket) = self.layers[layer_index].get_mut(&index) {
            if bucket.len() < self.slots[layer_index] {
                bucket.push(value);
            } else {
                // evict the oldest
                let mut oldest_index = 0;
                let mut oldest_timestamp = 0.0;
                // find the oldest entry
                for (i,slot) in bucket.iter().enumerate() {
                    if i == 0 {
                        oldest_timestamp = slot.1.0;
                        continue;
                    }
                    if slot.1.0 < oldest_timestamp {
                        oldest_timestamp = slot.1.0;
                        oldest_index = i;
                    }
                }
                // remove old value
                ret = Some(bucket.remove(oldest_index));
                self.evicted_cache_stats_vector.push((timestamp, Packet::from_binary_key_to_flow_id_string(ret.clone().unwrap().0), ret.clone().unwrap().2, layer_index+1, 0));
                // insert new one
                bucket.push(value);
            }
        } else {
            // easy folks!
            self.layers[layer_index].insert(index, vec![value]);
        }

        return ret;
    }

    fn find_update_remove_in_layer(&mut self, packet: &Packet, layer_index: usize) -> (bool, Option<(u128, (f64, Vec<Packet>), usize)>){
        assert!(layer_index < (self.packet_limit - 1));
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        layer_index.hash(&mut hasher);
        let key = packet.get_binary_key();
        key.hash(&mut hasher);
        let index = hasher.finish() as usize % self.rows[layer_index] ;
        let mut hit = false;
        let mut index_to_remove = 0;
        let mut ret = (false, None);
        if let Some(bucket) = self.layers[layer_index].get(&index) {
            for (i,slot) in bucket.iter().enumerate() {
                if slot.0 == key {
                    hit = true;
                    ret.0 = true;
                    index_to_remove = i;
                    break;
                }
            }
        }
        if hit {
            // Middle layer -> move to next
            if layer_index != (self.packet_limit - 2) {
                // remove the old slot
                let mut val = self.layers[layer_index].get_mut(&index).unwrap().remove(index_to_remove);
                val.1.0 = packet.timestamp;
                val.1.1.push(packet.clone());
                // we may get the evicted entry or None
                ret.1 = self.insert_into_layer(val, layer_index + 1, packet.timestamp);

            } else {
                // update the slot with new packet
                self.layers[layer_index].get_mut(&index).unwrap()[index_to_remove].1.1.push(packet.clone());
                self.layers[layer_index].get_mut(&index).unwrap()[index_to_remove].1.0 = packet.timestamp;
                ret.1 = Some(self.layers[layer_index].get_mut(&index).unwrap().remove(index_to_remove));
                self.collected_cache_stats_vector.push((packet.timestamp, Packet::from_binary_key_to_flow_id_string(key), self.packet_limit, ret.1.clone().unwrap().2, 0));
            }
        }
        return ret;
    }

}

impl FlowManager for HierarchicalFlowManager {
    fn dump_data(&mut self) -> HashMap<u128, usize> {
        let mut ret = HashMap::new();
        for layer in &mut self.layers {
            for bucket in layer {
                for slot in bucket.1 {
                    ret.insert(slot.0, slot.1.1.len());
                }
            }
        }
        ret
    }

    fn dump_raw_data(&mut self) -> HashMap<u128, Vec<Packet>> {
        let mut ret = HashMap::new();
        for layer in &mut self.layers {
            for bucket in layer {
                for slot in bucket.1 {
                    ret.insert(slot.0, slot.1.1.to_vec());
                }
            }
        }
        ret
    }

    fn set_evicted_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.evicted_receiver = Some(next_block);
    }

    fn set_model_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.model_receiver = Some(next_block);
    }

    fn get_received_flows_while_in_cache_collected(&mut self) -> Vec<(f64, String, usize, usize, usize)> {
        self.collected_cache_stats_vector.to_vec()
    }

    fn get_received_flows_while_in_cache_evicted(&mut self) -> Vec<(f64, String, usize, usize, usize)> {
        self.evicted_cache_stats_vector.to_vec()
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.evicted_cache_stats_vector.clear();
        self.collected_cache_stats_vector.clear();
        self.fake_syn_set.clear();
    }
}

impl PacketReceiver for HierarchicalFlowManager {

    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let mut temp = (false, None);
        let mut layer_index;
        let mut hit = false;
        for i in 0..(self.packet_limit-1) {
            layer_index = i;
            temp = self.find_update_remove_in_layer(packet, layer_index);
            if temp.0 {
                hit = true;
                break;
            }
        }
        // we hit, so there could be a packet to send to the cms (evicted or caching system ) or one to the model or none
        if hit {
            match temp.1 {
                Some(v) => {
                    // if the entry has < packet limit
                    if v.1.1.len() < self.packet_limit {
                        let temp_pkt = Packet::build_from_binary_key(v.0);
                        packet.clone_from_existing_packet(&temp_pkt);
                        packet.pkt_count = v.1.1.len();
                        packet.agg_features = prepare_aggregated_data(v.1.1);
                        packet.packet_source = PacketSource::FromFlowManager;
                        return self.evicted_receiver.clone();
                    } else if v.1.1.len() == self.packet_limit {
                        // send to the model
                        packet.agg_features = prepare_aggregated_data(v.1.1);
                        packet.pkt_count = self.packet_limit;
                        packet.packet_source = PacketSource::FromFlowManager;
                        return self.model_receiver.clone();
                    } else {
                        // shouldn't arrive here
                        assert!(0 == 1);
                    }
                },
                None => { return None},
            }
        } else {
            // insert into first layer
            if FAKE_SYN && self.fake_syn_set.contains(&packet.get_binary_key()) {
                packet.packet_source = PacketSource::FromFlowManager;
                return self.evicted_receiver.clone();
            } else {
                self.fake_syn_set.insert(packet.get_binary_key());
                let possibly_evicted = self.insert_into_layer((packet.get_binary_key(),(packet.timestamp, vec![packet.clone()]),0), 0, packet.timestamp);
                self.update_new_flow_counter(packet.get_binary_key());
                if let Some(v) = possibly_evicted {
                    // send value to the CMS
                    packet.clone_from_existing_packet(&v.1.1[0]);
                    packet.pkt_count = 1;
                    packet.packet_source = PacketSource::FromFlowManager;
                    packet.agg_features = prepare_aggregated_data(vec![packet.clone()]);
                    return self.evicted_receiver.clone();
                }
            }
        }

        return None;

    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

