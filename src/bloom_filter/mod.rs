use std::hash::Hash;
use std::hash::Hasher;
use crate::packet::PacketReceiver;
use crate::packet::PacketSource;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::cmp::Eq;
use crate::packet::Packet;

#[derive(Clone)]
pub struct BloomFilter<T: Hash + Eq + Copy> {
    filter_size: usize,
    hash_function_count: usize,
    filter_bins: Vec<bool>, 
    inserted_keys: HashSet<T>,
    false_positive_keys: HashSet<T>,
    false_positive_counter: usize,
}

pub struct MiceBloomFilter {
    bf: BloomFilter<u128>,
    mice_next_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
    next_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
    control_plane_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
}

impl MiceBloomFilter {
    pub fn build_mice_bloom_filter(filter_size: usize, hash_function_count: usize) -> MiceBloomFilter {
        let bf = BloomFilter::build_bloom_filter(filter_size, hash_function_count);
        MiceBloomFilter {
            bf,
            mice_next_block: None,
            next_block: None,
            control_plane_block: None,
        }
    }
    pub fn set_mice_next_block(&mut self, mice_next_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.mice_next_block = Some(mice_next_block);
    }

    pub fn set_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.next_block = Some(next_block);
    }

    pub fn set_control_plane_block(&mut self, control_plane_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.control_plane_block = Some(control_plane_block);
    }

    pub fn get_false_positive_count(&self) -> usize {
        return self.bf.false_positive_counter;
    }

    pub fn get_inserted_keys(&self) -> Vec<u128> {
        return self.bf.inserted_keys.clone().into_iter().collect::<Vec<u128>>();
    }

    pub fn get_false_positive_keys(&self) -> Vec<u128> {
        return self.bf.false_positive_keys.clone().into_iter().collect::<Vec<u128>>();
    }

}

impl PacketReceiver for MiceBloomFilter {
    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<std::rc::Rc<std::cell::RefCell<dyn PacketReceiver>>> {
        match packet.packet_source {
            PacketSource::FromModel | PacketSource::FromHashTableDiscarded => {
                packet.packet_source = PacketSource::FromBf;
                self.bf.insert(packet.get_binary_key());
                if let Some(v) = self.control_plane_block.clone() {
                    v.borrow_mut().receive_packet(packet);
                }
                self.mice_next_block.clone()
            },
            _ => {
                packet.packet_source = PacketSource::FromBf;
                if self.bf.key_is_present(packet.get_binary_key()) {
                    self.mice_next_block.clone()
                } else {
                    self.next_block.clone()
                }
            },
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
       self 
    }
}

impl <T: Hash + Eq + Copy> BloomFilter<T> {

    pub fn build_bloom_filter(filter_size: usize, hash_function_count: usize) -> BloomFilter<T> {
        let filter_bins = vec![false; filter_size];
        BloomFilter {
            filter_size, 
            hash_function_count,
            filter_bins,
            inserted_keys: HashSet::new(),
            false_positive_counter: 0,
            false_positive_keys: HashSet::new(),
        }
    }

    pub fn insert(&mut self, key: T) {
        let range = 0..self.hash_function_count;
        let hash_function_range = self.filter_size/self.hash_function_count; 
        for i in range {
            let mut hasher = std::collections::hash_map::DefaultHasher::default();
            i.hash(&mut hasher);
            key.hash(&mut hasher);
            let mut index = hasher.finish() as usize;
            index = index % hash_function_range;
            index = index + i*hash_function_range; 
            self.filter_bins[index] = true;
            self.inserted_keys.insert(key);
        }
    }

    pub fn key_is_present(&mut self, key: T) -> bool {
        let range = 0..self.hash_function_count;
        let hash_function_range = self.filter_size/self.hash_function_count; 
        let mut ret = true;
        for i in range {
            let mut hasher = std::collections::hash_map::DefaultHasher::default();
            i.hash(&mut hasher);
            key.hash(&mut hasher);
            let mut index = hasher.finish() as usize;
            index = index % hash_function_range;
            index = index + i*hash_function_range; 
            if self.filter_bins[index] == false {
                ret = false;
            }
        }
        if ret == true && !self.inserted_keys.contains(&key) {
            self.false_positive_counter += 1;
            self.false_positive_keys.insert(key);
        }
        return ret;
    }

    pub fn clear(&mut self) {
        for x in self.filter_bins.iter_mut() {
            *x = false;
        }
    }

}
