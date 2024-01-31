use std::hash::Hash;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use itertools::Itertools;

use crate::packet::PacketSource;
use super::packet::{Packet, PacketReceiver};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

const PROBABILITY_EVICTION: bool = false;
const ES_EVICTION: bool = false;


pub struct HashTable <K: Hash + PartialEq + Clone,V> {
    row_count: usize,
    slot_count: usize,
    tables: Vec<HashMap<usize,Vec<(K,V,u8)>>>, 
    table_count: usize,
    probability_table: HashMap<usize,Vec<f64>>,
    guard_value_table: HashMap<usize, usize>, 
    hash_seed: usize,
    next_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
    discarded_next_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
    discarded_keys: usize,
    discarded_keys_list: Vec<K>,
    rehit_keys: usize,
}

impl PacketReceiver for HashTable<u128,usize> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    // receive packet from the 
    fn receive_packet<'a>(&'a mut self, mut packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let key = packet.get_binary_key();
        match packet.packet_source {
            PacketSource::FromParser => { 
                let updated = self.update_key(key, packet.pkt_count);
                if updated {
                    //eprintln!("Hash Table HH hit");
                    return None;
                } else {
                   if let Some(_) = self.next_block.clone() {
                       packet.packet_source = PacketSource::FromHashTable;
                       return self.next_block.clone(); 
                   } else {
                       return None;
                   }
                }
            },
            PacketSource::FromModel => { 
                //let inserted = false;
                //if PROBABILITY_EVICTION {
                //    //inserted = self.insert_key_probability(key, packet.pkt_count, *packet.agg_features.get(&String::from("prediction_probability")).unwrap(), packet);
                //} else if ES_EVICTION {
                //    //inserted = self.insert_key_es_eviction(key, packet.pkt_count, packet);
                //} else {
                let inserted = self.insert_key(key, packet.pkt_count);
                //}
                if !inserted {
                    if let Some(_) = self.discarded_next_block.clone() {
                        packet.packet_source = PacketSource::FromHashTableDiscarded;
                        return self.discarded_next_block.clone();
                    }
                } 
                return None;
            },
            _ => {
                return None;
            }
        };
    }
}

impl HashTable<u128, usize> {
    pub fn dump_data_formatted(&self) -> Vec<String> {
        let mut ret = vec![];
        for (i, table) in self.tables.iter().enumerate() {
            for (_,v) in table {
                for i in v {
                    // return only hit flows
                    if i.2 == 1 {
                        let flow_id = Packet::from_binary_key_to_flow_id_string(i.0);
                        let size = format!("{}",i.1);
                        let concat = format!("{},{}", flow_id, size);
                        ret.push(concat);
                    }
                }
            }
        }
        ret
    }
    pub fn dump_data(&self) -> HashMap<u128,usize> {
        let mut ret = HashMap::new();
        for (i, table) in self.tables.iter().enumerate() {
            for (_,sl) in table {
                for (k,v,h) in sl {
                    if *h == 1 { 
                        ret.insert(*k, *v);
                    }
                }
            }
        }
        ret
    }

    //pub fn insert_key_probability(&mut self, key: u128, count: usize, probability: f64, packet: &mut Packet) -> bool {
    //    let key_index = self.get_key_index(&key);
    //    //println!("index {}", key_index);
    //    let mut ret = false;
    //    for table in &mut self.tables {
    //    }
    //    if let Some(v) = self.table.get_mut(&key_index) {
    //        if v.len() < self.slot_count {
    //            v.push((key, count, 1));
    //            self.probability_table.get_mut(&key_index).unwrap().push(probability);
    //            ret = true;
    //        } else {
    //            let prob_v = self.probability_table.get_mut(&key_index).unwrap();
    //            let mut min_prob = 1.0;
    //            let mut min_prob_index = 0;
    //            for (index,prob) in prob_v.iter().enumerate() {
    //                if index == 0 {
    //                    min_prob = *prob;
    //                    continue;
    //                }
    //                if *prob < min_prob {
    //                    min_prob = *prob;
    //                    min_prob_index = index;
    //                }
    //            }
    //            if min_prob < probability {
    //                let old_val = v.remove(min_prob_index);
    //                prob_v.remove(min_prob_index);
    //                v.push((key,count,1));
    //                prob_v.push(probability);
    //                let mut new_packet = Packet::build_from_binary_key(old_val.0);
    //                new_packet.pkt_count = old_val.1;
    //                packet.clone_from_existing_packet(&new_packet);
    //                // needed false to force forwarding to discarded sink (usually CMS)
    //                ret = false;
    //            }
    //        }
    //    } else {
    //        self.table.insert(key_index, vec![(key,count,1)]);
    //        self.probability_table.insert(key_index, vec![probability]);
    //        ret = true;
    //    }
    //    ret
    //}

    //pub fn insert_key_es_eviction(&mut self, key: u128, count: usize, packet: &mut Packet) -> bool {
    //    let key_index = self.get_key_index(&key);
    //    //println!("index {}", key_index);
    //    let mut ret = false;
    //    if let Some(v) = self.table.get_mut(&key_index) {
    //        if v.len() < self.slot_count {
    //            v.push((key, count, 1));
    //            ret = true;
    //        } else {
    //            let mut min_value = 0;
    //            let mut min_value_index = 0;
    //            for (index, entry) in v.iter().enumerate() {
    //                if index == 0 {
    //                    min_value = entry.1;
    //                    continue;
    //                }
    //                if entry.1 < min_value {
    //                    min_value = entry.1;
    //                    min_value_index = index;
    //                }
    //            }
    //            let guard_val = self.guard_value_table.get_mut(&key_index).unwrap();
    //            *guard_val += count;
    //            //if swap
    //            if *guard_val > min_value*8 {
    //                *guard_val = 0;
    //                let old_val = v.remove(min_value_index);
    //                v.push((key,count,1));
    //                let mut new_packet = Packet::build_from_binary_key(old_val.0);
    //                new_packet.pkt_count = old_val.1;
    //                packet.clone_from_existing_packet(&new_packet);
    //                // needed false to force forwarding to discarded sink (usually CMS)
    //                ret = false;
    //            } else {
    //                // send the flow that couldn't enter to CMS
    //                ret = false;
    //            }
    //        }
    //    } else {
    //        self.table.insert(key_index, vec![(key,count,1)]);
    //        self.guard_value_table.insert(key_index, 0);
    //        ret = true;
    //    }
    //    ret
    //}
}

impl <K: Hash + PartialEq + Clone> HashTable<K, usize> {
    pub fn build_hash_table(row_count: usize, slot_count: usize, hash_seed: usize, table_count: usize) -> HashTable<K,usize> {
        let tables = vec![HashMap::new(); table_count];
        HashTable {
            row_count,
            slot_count,
            tables,
            table_count,
            probability_table: HashMap::new(),
            guard_value_table: HashMap::new(),
            hash_seed,
            next_block: None,
            discarded_next_block: None,
            discarded_keys: 0,
            discarded_keys_list: vec![],
            rehit_keys: 0,
        }
    }

    pub fn get_load_factor(&self) -> f32 {
        let mut items = 0;
        for (i, table) in self.tables.iter().enumerate() {
            items += table.iter().map(|(k,v)| { v.len() }).reduce(|acc,e| { acc + e } ).unwrap_or(0);
        }
        (items as f32)/(self.row_count as f32 *self.slot_count as f32 * self.table_count as f32)
    }

    pub fn get_discarded_keys(&self) -> usize {
        self.discarded_keys
    }

    pub fn get_discarded_keys_list(&self) -> Vec<K>{
        self.discarded_keys_list.to_vec()
    }

    pub fn get_rehit_keys(&self) -> usize {
        self.rehit_keys
    }

    pub fn set_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>> ) {
        self.next_block = Some(next_block);
    }

    pub fn set_discarded_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>> ) {
        self.discarded_next_block = Some(next_block);
    }

    pub fn clear(&mut self) {
        for (i, table) in self.tables.iter_mut().enumerate() {
            table.iter_mut().foreach(|(_,v)| { v.clear();});
        }
    }

    pub fn delete_key(&mut self, key: K) {
        for (i, table) in self.tables.iter_mut().enumerate() {
            let key_index = Self::get_key_index(&key, i, self.hash_seed, self.row_count);
            if let Some(v) = table.get_mut(&key_index) {
                let mut temp = v.to_vec();
                for (i,(k,_,_)) in v.iter_mut().enumerate() { 
                    if *k == key {
                        temp.remove(i);
                    }
                }
                *v = temp;
            } 
        }
    }

    pub fn update_key(&mut self, key: K, count: usize) -> bool {
        let mut ret = false;
        for (i, table) in self.tables.iter_mut().enumerate() {
            let key_index = Self::get_key_index(&key, i, self.hash_seed, self.row_count);
            if let Some(v) = table.get_mut(&key_index) {
                for (k,c,b) in v { 
                    if *k == key {
                        *c += count;
                        ret = true;
                        // set again unmutable bit, correct strategy?
                        if *b == 0 {
                            self.rehit_keys += 1;
                            *c = 1; 
                        }
                        *b = 1;
                        break;
                    }
                }
            } 
        }
        ret
    }
    
    pub fn insert_key(&mut self, key: K, count: usize) -> bool {
        let mut ret = false;
        for (i, table) in self.tables.iter_mut().enumerate() {
            let key_index = Self::get_key_index(&key, i, self.hash_seed, self.row_count);
            //println!("index {}", key_index);
            if let Some(v) = table.get_mut(&key_index) {
                if v.len() < self.slot_count {
                    v.push((key, count, 1));
                    ret = true;
                    break;
                } else {
                    if v[0].2 == 0 {
                        v.remove(0);
                        v.push((key, count, 1));
                        ret = true;
                        break;
                    } else {
                        if i == (self.table_count - 1) { 
                            self.discarded_keys += 1;
                            self.discarded_keys_list.push(key.clone());
                        }
                    }
                }
            } else {
                table.insert(key_index, vec![(key,count,1)]);
                ret = true;
                break;
            }
        }
        ret
    }

    pub fn get_key_value(&self, key: &K) -> Option<usize> {
        for (i, table) in self.tables.iter().enumerate() {
            let index = Self::get_key_index(key, i, self.hash_seed, self.row_count);
            if let Some(v) = table.get(&index) {
                for (k,val,_) in v {
                    if k == key {
                        return Some(*val);
                    }
                }
            } 
        }
        return None;
    }

    fn get_key_index(key: &K, table_index: usize, hash_seed: usize, row_count: usize) -> usize {
        let mut hasher = DefaultHasher::default();
        hash_seed.hash(&mut hasher);
        table_index.hash(&mut hasher);
        key.hash(&mut hasher);
        let index = (hasher.finish() as usize ) % row_count;
        index
    }

    fn reset_unmutable_bit(&mut self) {
        for (_, table) in self.tables.iter_mut().enumerate() {
            table.iter_mut().foreach(|(_,vector)| {vector.iter_mut().foreach(|(_,_,bit)| {*bit = 0} )});
        }
    }

    pub fn new_epoch(&mut self) {
        self.reset_unmutable_bit();
        self.rehit_keys = 0;
        self.discarded_keys = 0;
    }
}
