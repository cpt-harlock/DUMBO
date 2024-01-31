use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use crate::hash_table::HashTable;
use crate::packet::{Packet, PacketReceiver, PacketSource};
use std::time::Instant;
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::cell::RefCell;
use std::iter::Iterator;
use crate::flow_manager::{FAKE_SYN, FlowManagerInnerStruct, FlowManager, prepare_aggregated_data};

pub struct InfiniteFlowManagerInnerStruct {
    // HashMap key: row index
    // HashMap value: vector of (key, (last_timestamp, vector of packets), new flows arrived while in the cache before collecting K pkts)
    hash_table: HashMap<u128, (f64, Vec<Packet>, usize)>,
    evicted_receiver: Option<Rc<RefCell<dyn PacketReceiver>>>,
    model_receiver: Option<Rc<RefCell<dyn PacketReceiver>>>,
    collected_cache_stats_vector: Vec<(f64,String,usize,usize,usize)>,
    packet_limit: usize,
}

pub struct InfiniteFlowManager {
    inner_struct: InfiniteFlowManagerInnerStruct,
}

impl InfiniteFlowManager {
    pub fn build(packet_limit: usize) -> InfiniteFlowManager {
        InfiniteFlowManager {
            inner_struct: InfiniteFlowManagerInnerStruct {
                hash_table: HashMap::new(),
                evicted_receiver: None,
                model_receiver: None,
                collected_cache_stats_vector: vec![],
                packet_limit
            }
        }
    }

    pub fn get_unprocessed_flows(&self) -> Vec<Packet> {
        let mut ret = vec![];
        for (k,v) in &self.inner_struct.hash_table {
            if v.1.len() < self.inner_struct.packet_limit {
                let mut pkt = Packet::build_from_binary_key(*k);
                pkt.pkt_count = v.1.len();
                ret.push(pkt);
            }
        }
        ret
    }
}

impl FlowManager for InfiniteFlowManager {
    fn dump_data(&mut self) -> HashMap<u128, usize> {
        let mut ret = HashMap::new();
        for (k,v) in &self.inner_struct.hash_table {
            ret.insert(*k, v.1.len());
        }
        ret
    }

    fn dump_raw_data(&mut self) -> HashMap<u128, Vec<Packet>> {
        let mut ret = HashMap::new();
        for (k,v) in &self.inner_struct.hash_table {
            ret.insert(*k, v.1.to_vec());
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
        vec![]
    }

    fn clear(&mut self) {
        self.inner_struct.hash_table.clear();
        self.inner_struct.collected_cache_stats_vector.clear();
    }
}

impl PacketReceiver for InfiniteFlowManager {

    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let inner_struct = &mut self.inner_struct;
        let flow_key = packet.get_binary_key();
        let ret; // = (packet, None);
        let packet_limit = inner_struct.packet_limit;
        // 0 -> None, 1 -> model_receiver, 2 -> evicted_receiver
        let mut destination_selector = 0;
        let mut new_flow = false;
        // if bucket already existing
        if let Some(bucket) = inner_struct.hash_table.get_mut(&flow_key) {
            // push to flow manager packet list
            bucket.1.push(packet.clone());
            bucket.0  = packet.timestamp;
            // send to model only when reaching the packet limit
            if bucket.1.len() == packet_limit {
                let agg_feat = prepare_aggregated_data(bucket.1.to_vec());
                packet.agg_features = agg_feat;
                packet.pkt_count = packet_limit;
                // sent to model
                destination_selector = 1;
                inner_struct.collected_cache_stats_vector.push((packet.timestamp,Packet::from_binary_key_to_flow_id_string(packet.get_binary_key()),bucket.2, bucket.1.len(), destination_selector));
                //remove_index = i;
                //remove_flag = true;
            } else if bucket.1.len() > packet_limit {
                // do not remove, send a one packet increase to the cms
                // will it work? many issues here...
                destination_selector = 2;
            } else {
                destination_selector = 0;
            }
        } else {
            inner_struct.hash_table.insert(flow_key, (packet.timestamp, vec![packet.clone()], 0));
            new_flow = true;
        }
        // if a new flow
        if new_flow {
            for tmp in &mut inner_struct.hash_table {
                if *tmp.0 != flow_key {
                    tmp.1.2 += 1;
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
