pub mod fm_evict_oldest;
pub mod fm_evict_timeout;
pub mod fm_infinite;
pub mod fm_hierarchical;

use std::collections::{HashMap, HashSet};
use crate::packet::{Packet, PacketReceiver};
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::cell::RefCell;
use std::iter::Iterator;

const FAKE_SYN: bool = true;

fn prepare_aggregated_data(packet_vector: Vec<Packet>) -> HashMap<String,f64> {
    //TODO: separate different stats into functions
    let mut ret = HashMap::new();
    let mut avg_size = 0.0;
    let mut stdev_size = 0.0;
    let mut avg_iat = 0.0;
    let mut stdev_iat = 0.0;
    let mut iat = 0.0;
    let mut timestamp_list = vec![];
    if packet_vector[0].agg_features.contains_key("Size") {
        for (i,p) in packet_vector.to_vec().into_iter().enumerate() {
            avg_size += p.agg_features.get("Size").unwrap();
            timestamp_list.push(p.timestamp);
            if i == 0 {
                iat = p.timestamp;
            } else {
                avg_iat += p.timestamp - iat;
                iat = p.timestamp;
            }
        }
        avg_size = avg_size / packet_vector.len() as f64;
        if packet_vector.len() > 1  {
            avg_iat = avg_iat / (packet_vector.len() - 1) as f64;
        }
        for (i, p) in packet_vector.to_vec().into_iter().enumerate() {
            stdev_size += (avg_size - p.agg_features.get("Size").unwrap()).powf(2.0);
            if i == 0 {
                iat = p.timestamp;
            } else {
                stdev_iat += (avg_iat - (p.timestamp as f64 - iat)).powf(2.0);
                iat = p.timestamp;
            }
        }
        stdev_size = stdev_size / packet_vector.len() as f64;
        if packet_vector.len() > 1  {
            stdev_iat = stdev_iat / (packet_vector.len() - 1) as f64;
        }
        stdev_size = stdev_size.sqrt();
        stdev_iat = stdev_iat.sqrt();
    }

    ret.insert(String::from("avg_size"), avg_size);
    ret.insert(String::from("stdev_size"), stdev_size);
    ret.insert(String::from("avg_iat"), avg_iat);
    ret.insert(String::from("stdev_iat"), stdev_iat);
    for (i, tmsp) in timestamp_list.iter().enumerate() {
        ret.insert(format!("timestamp_{}",i), *tmsp);
    }
    ret
}

pub struct FlowManagerInnerStruct {
    hash_table: HashMap<usize, Vec<(u128,(f64, Vec<Packet>), usize)>>,
    rows: usize,
    slots:  usize,
    seed: usize,
    evicted_receiver: Option<Rc<RefCell<dyn PacketReceiver>>>,
    model_receiver: Option<Rc<RefCell<dyn PacketReceiver>>>,
    evicted_cache_stats_vector: Vec<(f64,String,usize,usize,usize)>,
    collected_cache_stats_vector: Vec<(f64,String,usize,usize,usize)>,
    packet_limit: usize,
    fake_syn_set: HashSet<u128>,
}

pub trait FlowManager: PacketReceiver {

    fn dump_data(&mut self) -> HashMap<u128,usize>;

    fn dump_raw_data(&mut self) -> HashMap<u128,Vec<Packet>>;

    fn set_evicted_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>);

    fn set_model_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>);

    fn get_received_flows_while_in_cache_collected(&mut self) -> Vec<(f64,String,usize,usize,usize)>;

    fn get_received_flows_while_in_cache_evicted(&mut self) -> Vec<(f64,String,usize,usize,usize)>;

    fn clear(&mut self);
}
