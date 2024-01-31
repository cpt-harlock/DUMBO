use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use crate::packet::{Packet, PacketReceiver, PacketSource};
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::cell::RefCell;
use std::iter::Iterator;
use crate::flow_manager::{FAKE_SYN, FlowManagerInnerStruct, FlowManager, prepare_aggregated_data};


pub struct FlowManagerEvictOldest {
    inner_struct: FlowManagerInnerStruct,
}

impl FlowManagerEvictOldest {
    pub fn build(rows: usize, slots: usize, packet_limit: usize, seed: usize) -> FlowManagerEvictOldest {
        FlowManagerEvictOldest {
            inner_struct: FlowManagerInnerStruct {
                hash_table: HashMap::new(),
                rows,
                slots,
                seed,
                evicted_receiver: None,
                model_receiver: None,
                evicted_cache_stats_vector: vec![],
                collected_cache_stats_vector: vec![],
                packet_limit,
                fake_syn_set: HashSet::new(),
            },
        }
    }

    fn evict_old_flow(&mut self, _: Packet, _: u128) -> bool {
        true
    }

    fn get_fm_inner_struct(&mut self) -> &mut FlowManagerInnerStruct {
        &mut self.inner_struct
    }
}

impl FlowManager for FlowManagerEvictOldest {

    fn dump_data(&mut self) -> HashMap<u128, usize> {
        let mut ret = HashMap::new();
        for (_,v) in &self.inner_struct.hash_table {
            for f in v {
                ret.insert(f.0, f.1.1.len());
            }
        }
        ret
    }

    fn dump_raw_data(&mut self) -> HashMap<u128, Vec<Packet>> {
        let mut ret = HashMap::new();
        for (_,v) in &self.inner_struct.hash_table {
            for f in v {
                ret.insert(f.0, f.1.1.to_vec());
            }
        }
        ret
    }

    fn set_evicted_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.inner_struct.evicted_receiver = Some(next_block);
    }

    fn set_model_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.inner_struct.model_receiver = Some(next_block);
    }

    fn get_received_flows_while_in_cache_collected(&mut self) -> Vec<(f64, String, usize, usize, usize)> {
        self.inner_struct.collected_cache_stats_vector.to_vec()
    }

    fn get_received_flows_while_in_cache_evicted(&mut self) -> Vec<(f64, String, usize, usize, usize)> {
        self.inner_struct.evicted_cache_stats_vector.to_vec()
    }

    fn clear(&mut self) {
        self.inner_struct.hash_table.clear();
        self.inner_struct.evicted_cache_stats_vector.clear();
        self.inner_struct.collected_cache_stats_vector.clear();
        self.inner_struct.fake_syn_set.clear();
    }
}

impl PacketReceiver for FlowManagerEvictOldest {

    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let inner_struct = self.get_fm_inner_struct();
        let mut hasher = DefaultHasher::new();
        inner_struct.seed.hash(&mut hasher);
        let flow_key = packet.get_binary_key();
        flow_key.hash(&mut hasher);
        let ret; // = (packet, None);
        let index = (hasher.finish() % inner_struct.rows as u64) as usize;
        let mut hit: bool = false;
        let mut oldest_index: usize = 0;
        let mut oldest_value = 0.0;
        let packet_limit = inner_struct.packet_limit;
        // 0 -> None, 1 -> model_receiver, 2 -> evicted_receiver
        let mut destination_selector = 0;
        // if bucket already existing
        if let Some(bucket) = inner_struct.hash_table.get_mut(&index) {
            for (i, slot) in bucket.iter_mut().enumerate()  {
                //identify the oldest value in the bucket
                if i == 0 {
                    oldest_value = packet.timestamp;
                    oldest_index = 0;
                } else {
                    // save the index of the oldest flow in the cache
                    if packet.timestamp < oldest_value {
                        oldest_value = packet.timestamp;
                        oldest_index = i;
                    }
                }
                if slot.0 == flow_key {
                    hit = true;
                    // push to Flow Manager packet list
                    slot.1.1.push(packet.clone());
                    slot.1.0  = packet.timestamp;
                    // send to model only when reaching the packet limit
                    if slot.1.1.len() == packet_limit {
                        let agg_feat = prepare_aggregated_data(slot.1.1.to_vec());
                        packet.agg_features = agg_feat;
                        packet.pkt_count = packet_limit;
                        // send to model
                        destination_selector = 1;
                        inner_struct.collected_cache_stats_vector.push((packet.timestamp,Packet::from_binary_key_to_flow_id_string(packet.get_binary_key()),slot.2, slot.1.1.len(), destination_selector));
                        //remove_index = i;
                        //remove_flag = true;
                        if FAKE_SYN {
                            // if it is predicted as an HH and inserted, this entry is useless
                            // instead, if a mice or discarded by hash table, the fake syn
                            // mechanism will send anyway all the packets to the CMS
                            bucket.remove(i);
                        }

                    } else if slot.1.1.len() > packet_limit {
                        // do not remove, send a one packet increase to the cms
                        // will it work? many issues here...
                        destination_selector = 2;
                    } else {
                        destination_selector = 0;
                    }
                    break;
                }
            }
            // we hit and we need to remove
            //if remove_flag {
            //    // push number of other flows that entered the cache while being here
            //    inner_struct.cache_stats_vector.push((packet.timestamp, bucket.get(remove_index).unwrap().2));
            //    bucket.remove(remove_index);
            //}
            // already returned if there was an hit -> here if no hit
            // if another free slot
            if !hit {
                // if the flow is not already inside the Flow Manager and we already meet it
                // (as if every flow starts with a SYN), send directly to the CMS
                if FAKE_SYN && inner_struct.fake_syn_set.contains(&flow_key) {
                    // send directly to cms
                    destination_selector = 2;
                } else {
                    // save the flow SYN
                    inner_struct.fake_syn_set.insert(flow_key);
                    // new flow is always saved
                    bucket.push((flow_key, (packet.timestamp, vec![packet.clone()]), 0));
                    // remove oldest flow in case it's required
                    if bucket.len() > inner_struct.slots {
                        //inner_struct.cache_stats_vector.push(bucket.get(oldest_index).unwrap().2);
                        let old = bucket.remove(oldest_index);
                        let mut pkt_to_ret = Packet::build_from_binary_key(old.1.1.get(0).unwrap().get_binary_key());
                        destination_selector = 2;
                        pkt_to_ret.pkt_count = old.1.1.len();
                        // in case we used the inference cache mechanism, we must avoid sending a packet
                        if old.1.1.len() >= packet_limit {
                            destination_selector = 0;
                        }
                        // due to Flow Manager functioning, we already accounted for this packet in cms
                        inner_struct.evicted_cache_stats_vector.push((packet.timestamp,Packet::from_binary_key_to_flow_id_string(old.0),old.2, old.1.1.len(), destination_selector));
                        pkt_to_ret.agg_features = prepare_aggregated_data(old.1.1);
                        packet.clone_from_existing_packet(&pkt_to_ret);
                    }
                    for tmp in &mut inner_struct.hash_table {
                        for fl in tmp.1 {
                            if fl.0 != flow_key {
                                fl.2 += 1;
                            }
                        }
                    }
                }
            }
        } else {
            inner_struct.hash_table.insert(index, vec![(flow_key,(packet.timestamp, vec![packet.clone()]), 0)]);
            inner_struct.fake_syn_set.insert(flow_key);
            for tmp in &mut inner_struct.hash_table {
                for fl in tmp.1 {
                    if fl.0 != flow_key {
                        fl.2 += 1;
                    }
                }
            }
        }
        if destination_selector == 0 {
            ret = None;
        } else if destination_selector == 1 {
            ret = inner_struct.model_receiver.clone();
        } else {
            ret = inner_struct.evicted_receiver.clone();
        }

        // setting packet source
        packet.packet_source = PacketSource::FromFlowManager;
        return ret;

    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
