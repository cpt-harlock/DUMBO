use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use crate::packet::PacketSource;
use blake3::Hasher as Blake3Hasher;
use rand_distr::num_traits::ToBytes;


use super::packet::{PacketReceiver, Packet};
use std::any::Any;
use std::rc::Rc;
use std::cell::RefCell;
use std::iter::Iterator;

/// Count Min-Sketch
pub struct CountMinSketch {
    pub bins: Vec<Vec<usize>>,
    rows: usize,
    columns: usize,
    bin_size: usize,
    bin_zero_size: usize,
    control_plane_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
    next_block_as_bf: Option<Rc<RefCell<dyn PacketReceiver>>>,
}

pub trait IntoBytes: Sized {
    fn my_to_le_bytes(&self) -> Vec<u8>;
}

impl IntoBytes for u128 {
    fn my_to_le_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl<T> IntoBytes for &T
    where
        T: IntoBytes,
{
    fn my_to_le_bytes(&self) -> Vec<u8> {
        return (**self).my_to_le_bytes();
    }
}

impl CountMinSketch {
    pub fn build_cms(rows: usize, columns: usize, bin_size: usize, bin_zero_size: usize) -> CountMinSketch {
        println!("{} {}", rows, columns);
        //need to check that I can actually build #rows hash function of #columns range
        assert!(columns as f64 <= 2.0f64.powf(64.0));
        let bins = vec![vec![0; columns]; rows];
        CountMinSketch {
            bins,
            rows,
            columns,
            bin_size,
            bin_zero_size,
            control_plane_block: None,
            next_block_as_bf: None,
        }
    }

    pub fn get_used_columns(&self) -> f64 {
        let mut filter = Vec::<u32>::new();
        for _ in 0..self.columns {
            filter.push(0);
        }
        for row in &self.bins {
            for (i,bin) in row.iter().enumerate() {
                if *bin != 0 {
                    filter[i] = 1;
                }
            }
        }
        let mut counter = 0;
        for bin in &filter {
            if *bin != 0 {
                counter += 1;
            }
        }
        counter as f64/self.columns as f64
    }

    pub fn get_load_factor(&self) -> f64 {
        let mut ret = 0.0;
        for row in &self.bins {
            for bin in row {
                if *bin != 0 {
                    ret += 1.0;
                }
            }
        }
        ret/((self.rows * self.columns) as f64)
    }

    pub fn bloom_filter<T: Hash + IntoBytes>(&mut self, value: T) -> bool {
        //eprintln!("CMS Bloom Filter");
        let mut ret = true;
        let column_indices = self.get_colums_index(value);
        for (i,v) in self.bins.iter_mut().enumerate() {
            let j = column_indices[i];
            //eprintln!("CMS Bloom Filter Value {}", v[j]);
            if v[j] == 0 {
                //eprintln!("AHAHAHAHAHA");
                ret = false;
                break;
            }
        }
        ret
    }

    pub fn set_control_plane_block(&mut self, control_plane_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.control_plane_block = Some(control_plane_block);
    }

    pub fn set_next_block_as_bf(&mut self, next_block_as_bf: Rc<RefCell<dyn PacketReceiver>>) {
        self.next_block_as_bf = Some(next_block_as_bf);
    }

    pub fn insert_element<T: Hash+IntoBytes>(&mut self, value: T, count: usize) {
        let column_indices = self.get_colums_index(value);
        for (i,v) in self.bins.iter_mut().enumerate() {
            let j = column_indices[i];
            v[j] += count;
            if (i==0) && v[j] > (2u64.pow(self.bin_zero_size as u32) - 1) as usize {
                v[j] = (2u64.pow(self.bin_zero_size as u32) - 1) as usize;
            }
            if (i>0) && v[j] > (2u64.pow(self.bin_size as u32) - 1) as usize {
                v[j] = (2u64.pow(self.bin_size as u32) - 1)  as usize;
            }
        }
    }

    pub fn insert_data<T: Hash+IntoBytes>(&mut self, data: &[T]) {
        data.iter()
            .map(|x| {
                let v = self.get_colums_index(x);
                for (i, j) in v.into_iter().enumerate() {
                    self.bins[i][j] += 1;
                    if (i==0) && self.bins[i][j] > (2u64.pow(self.bin_zero_size as u32) - 1) as usize {
                        self.bins[i][j] = (2u64.pow(self.bin_zero_size as u32) - 1) as usize;
                    }
                    if (i>0) && self.bins[i][j] > (2u64.pow(self.bin_size as u32) - 1) as usize {
                        self.bins[i][j] = (2u64.pow(self.bin_size as u32) - 1)  as usize;
                    }
                }
            })
            .for_each(drop);
    }

    pub fn get_size_estimate<T: Hash + IntoBytes>(&self, value: T) -> usize {
        let column_indices = self.get_colums_index(value);
        let mut ret_val = std::usize::MAX;
        //let mut ret_val = 2u32.pow(self.bin_size as u32) as usize - 1 ;
        for (i, x) in column_indices.into_iter().enumerate() {
            if (i>0) && (self.bins[i][x] == (2u64.pow(self.bin_size as u32) - 1) as usize) {
                continue;
            }
            //if (i==0) && (self.bins[i][x] == (2u64.pow(self.bin_zero_size as u32) - 1) as usize) {
            //    continue;
            //}
            if self.bins[i][x] < ret_val {
                ret_val = self.bins[i][x];
            }
        }
        ret_val
    }

    pub fn get_colums_index<T: Hash + IntoBytes>(&self, value: T) -> Vec<usize> {
        let mut ret_vec = Vec::<usize>::new();
        for i in 0..self.rows {
            // let mut hasher = DefaultHasher::new();
            // i.hash(&mut hasher);
            // value.hash(&mut hasher);
            // let hash = hasher.finish();
            // let column_index = hash % self.columns as u64;
            let mut hasher = Blake3Hasher::new();
            hasher.update(&i.to_ne_bytes());
            hasher.update(&value.my_to_le_bytes());
            let hash = u128::from_str_radix(&hasher.finalize().to_string()[0..8], 16).unwrap();
            let column_index = hash % self.columns as u128;
            ret_vec.push(column_index as usize);
        }
        ret_vec
    }

    pub fn clear(&mut self) {
        self.bins.iter_mut().for_each(|v| {v.iter_mut().for_each(|x| {*x = 0;})});
    }
}

impl PacketReceiver for CountMinSketch {
    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        match packet.packet_source {
            // in this case it's used as a Bloom Filter
            PacketSource::FromHashTable => {
                //eprintln!("RECEIVED FROM HASH TABLE");
                let bf_res = self.bloom_filter(packet.get_binary_key());
                if bf_res {
                    self.insert_element(packet.get_binary_key(), packet.pkt_count);
                    if let Some(v) = self.control_plane_block.clone() {
                        v.borrow_mut().receive_packet(packet);
                    }
                } else {
                    return self.next_block_as_bf.clone();
                }

            },
            _ => {
                //eprintln!("RECEIVED NORMALLY");
                self.insert_element(packet.get_binary_key(), packet.pkt_count);
                if let Some(v) = self.control_plane_block.clone() {
                    v.borrow_mut().receive_packet(packet);
                }
            }
        }
        return None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
