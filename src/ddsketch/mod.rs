mod mapping;
use crate::packet::{PacketReceiver, Packet, PacketSource};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::cell::RefCell;

pub struct DDSketchIAT {
    pub ddsketch: DDSketch,
    last_timestamp: f64,
}

pub struct DDSketchArray {
    pub ddsketch_array: HashMap<u128, DDSketchIAT>,
    max_iat: f64,
    min_iat: f64,
    bins: usize,
    bin_size: usize,
}

impl DDSketchArray {
    pub fn build_ddsketch_array(max_iat: f64, min_iat: f64, bins: usize, bin_size: usize) -> DDSketchArray {
        DDSketchArray { ddsketch_array: HashMap::new(), max_iat, min_iat, bins, bin_size}
    }
}

impl PacketReceiver for DDSketchArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let key = packet.get_binary_key();
        if let Some(ddsketch) = self.ddsketch_array.get_mut(&key) {
            ddsketch.insert_value(packet.timestamp);
        } else {
            // create ddsketch
            self.ddsketch_array.insert(key, DDSketchIAT { ddsketch: DDSketch::build_ddsketch(self.max_iat, self.min_iat, self.bins, self.max_iat, self.min_iat, 0, 0, self.bin_size) , last_timestamp: 0.0 });
            //println!("inserting key {}", Packet::from_binary_key_to_flow_id_string(key));
            // get value array
            let mut timestamp_vec: Vec<f64> = vec![];
            for i in 0..5 {
                let val = *packet.agg_features.get(format!("timestamp_{}",i).as_str()).unwrap_or(&0.0);
                if val == 0.0 {
                    break;
                } else {
                    timestamp_vec.push(val);
                }
            }
            let ddsketch = self.ddsketch_array.get_mut(&key).unwrap();
            for (i, v) in timestamp_vec.into_iter().enumerate() {
                if i == 0 {
                    ddsketch.set_timestamp(v);
                } else {
                    ddsketch.insert_value(v);
                }
            }

        }
        None
    }
}

#[derive(Debug)]
pub struct DDSketch {
    pub central_mapping: mapping::LogMaxMinMapping,
    pub central_bins: Vec<u32>,
    pub left_margin_mapping: Option<mapping::LogMaxMinMapping>,
    pub left_bins: Vec<u32>,
    pub right_margin_mapping: Option<mapping::LogMaxMinMapping>,
    pub right_bins: Vec<u32>,
    pub global_min: f64,
    pub global_max: f64,
    pub central_min: f64,
    pub central_max: f64,
    bin_size: usize,
    zero_count: usize,
    unfreezed: bool,
}

pub struct DDSketchFilter {
    hh_ddsketch: Option<Rc<RefCell<dyn PacketReceiver>>>,
    mice_ddsketch: Option<Rc<RefCell<dyn PacketReceiver>>>,
    flow_manager: Option<Rc<RefCell<dyn PacketReceiver>>>,
    pub hh_ddsketch_set: HashSet<u128>,
    pub mice_ddsketch_set: HashSet<u128>,
    max_hh: usize,
}

impl DDSketchFilter {
    pub fn build_ddsketch_filter(max_hh: usize) -> DDSketchFilter {
        DDSketchFilter { hh_ddsketch: None, mice_ddsketch: None, flow_manager: None, hh_ddsketch_set: HashSet::new(), mice_ddsketch_set: HashSet::new(), max_hh }
    }

    pub fn set_hh_ddsketch(&mut self, block: Rc<RefCell<dyn PacketReceiver>>) {
        self.hh_ddsketch = Some(block);
    }

    pub fn set_mice_ddsketch(&mut self, block: Rc<RefCell<dyn PacketReceiver>>) {
        self.mice_ddsketch= Some(block);
    }

    pub fn set_flow_manager(&mut self, block: Rc<RefCell<dyn PacketReceiver>>) {
        self.flow_manager = Some(block);
    }
}

impl PacketReceiver for DDSketchFilter {

    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let key = packet.get_binary_key();
        match packet.packet_source {
            PacketSource::FromParser => {
                if self.hh_ddsketch_set.contains(&key) {
                    return self.hh_ddsketch.clone();
                } else if self.mice_ddsketch_set.contains(&key) {
                    return self.mice_ddsketch.clone();
                } else {
                    return self.flow_manager.clone();
                }
            },
            PacketSource::FromModel => {
                if *packet.agg_features.get("label").unwrap() == 1.0 && self.hh_ddsketch_set.len() <= self.max_hh {
                    self.hh_ddsketch_set.insert(key);
                    return self.hh_ddsketch.clone();
                } else {
                    //println!("received mice");
                    self.mice_ddsketch_set.insert(key);
                    return self.mice_ddsketch.clone();
                }
            },
            PacketSource::FromFlowManager => {
                //println!("received mice with key {}", Packet::from_binary_key_to_flow_id_string(key));
                self.mice_ddsketch_set.insert(key);
                return self.mice_ddsketch.clone();
            }
            _ => {
                println!("unknown source");
                println!("{:?}", packet.packet_source);
                std::process::exit(-1)
            },
        };
        None 
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self 
    }
}

impl DDSketchIAT {
    fn insert_value(&mut self, timestamp: f64) {
        // insert into DDSketch and done :)
        let iat = timestamp - self.last_timestamp;
        self.last_timestamp = timestamp;
        self.ddsketch.insert_values_array(&vec![iat]);
    }

    fn set_timestamp(&mut self, timestamp: f64) {
        self.last_timestamp = timestamp;
    }

}

impl DDSketch {
    pub fn build_ddsketch(
        max: f64,
        min: f64,
        bins_count: usize,
        global_max: f64,
        global_min: f64,
        left_margin_bins_count: usize,
        right_margin_bins_count: usize,
        bin_size: usize
    ) -> DDSketch {
        //println!("ddsketch max min: {} {}", max, min);
        let left_margin_mapping;
        let left_bins;
        if left_margin_bins_count > 0 && (min - global_min) > 0.0 {
            left_margin_mapping = Some(mapping::LogMaxMinMapping::build_mapping(
                min,
                global_min,
                left_margin_bins_count,
            ));
            left_bins = vec![0; left_margin_bins_count];
        } else {
            left_margin_mapping = None;
            left_bins = Vec::<_>::new();
        }
        let right_margin_mapping;
        let right_bins;
        if right_margin_bins_count > 0 && (global_max - max) > 0.0 {
            right_margin_mapping = Some(mapping::LogMaxMinMapping::build_mapping(
                global_max,
                max,
                right_margin_bins_count,
            ));
            right_bins = vec![0; right_margin_bins_count];
        } else {
            right_margin_mapping = None;
            right_bins = Vec::<_>::new();
        }
        let d = DDSketch {
            central_mapping: mapping::LogMaxMinMapping::build_mapping(max, min, bins_count),
            central_bins: vec![0; bins_count],
            left_margin_mapping: left_margin_mapping,
            left_bins: left_bins,
            right_margin_mapping: right_margin_mapping,
            right_bins: right_bins,
            zero_count: 0,
            central_max: max,
            central_min: min,
            global_max: global_max,
            global_min: global_min,
            bin_size: bin_size,
            unfreezed: true,
        };
        d
    }

    pub fn insert_values_array(&mut self, value_array: &Vec<f64>) {
        if self.unfreezed {
            let mut temp_vector = value_array.to_vec();
        for x in &temp_vector {
            if *x < self.global_min {
                self.zero_count += 1;
            }
        }
        //remove 0.0f64 elements
        temp_vector.retain(|x| *x >= self.global_min);
        //println!("vector after != 0"); 
        //println!("{:?}", temp_vector);
        for x in temp_vector {
            //left bins
            if (self.central_min != self.global_min)
                && (self.left_bins.len() > 0)
                && (x >= self.global_min)
                && (x <= self.central_min)
            {
                //println!("should insert in left bins");
                if let Some(mapp) = &self.left_margin_mapping {
                    self.left_bins[mapp.val_to_bin(x)] += 1;
                    //println!("insert in left bins {}", x);
                    continue;
                }
            }
            //right bins
            if (self.central_max != self.global_max)
                && (self.right_bins.len() > 0)
                && (x >= self.central_max)
                && (x <= self.global_max)
            {
                //println!("should insert in right bins");
                if let Some(mapp) = &self.right_margin_mapping {
                    //println!("insert in right bins {}", x);
                    self.right_bins[mapp.val_to_bin(x)] += 1;
                    continue;
                }
            }

            //println!("insert in central bins {}", x);
            self.central_bins[self.central_mapping.val_to_bin(x)] += 1;
        }

        for x in &mut self.left_bins {
            if *x >= 2u32.pow(self.bin_size as u32) - 1 {
                self.unfreezed = false;
            }
            *x = std::cmp::min(*x, 2u32.pow(self.bin_size as u32) - 1);
        }
        for x in &mut self.central_bins {
            if *x >= 2u32.pow(self.bin_size as u32) - 1 {
                self.unfreezed = false;
            }
            *x = std::cmp::min(*x, 2u32.pow(self.bin_size as u32) - 1);
        }
        for x in &mut self.right_bins {
            if *x >= 2u32.pow(self.bin_size as u32) - 1 {
                self.unfreezed = false;
            }
            *x = std::cmp::min(*x, 2u32.pow(self.bin_size as u32) - 1);
        }
        //println!("{:?}", self.bins);
        //println!("{:?}", self.zero_count);
        }
        
    }

    pub fn get_used_bin_indices(&self) -> Vec<usize> {
        let mut temp = self.central_bins.iter().enumerate().collect::<Vec<_>>();
        temp.retain(|(index, value)| **value != 0);
        let ret = temp.iter().map(|(index,value)| *index).collect::<Vec<_>>();
        ret
    }
    pub fn get_sparsity(&self) -> (usize, usize) {
        let s = self
            .central_bins
            .iter()
            .map(|x| {
                (if *x != 0 as u32 {
                    return 1;
                } else {
                    return 0;
                })
            })
            .collect::<Vec<usize>>()
            .iter()
            .sum();
        (s, self.central_bins.capacity())
    }

    pub fn get_quantile(&self, quantile: f64) -> f64 {
        let limit = quantile
            * (((self.central_bins.iter().sum::<u32>()
                + self.left_bins.iter().sum::<u32>()
                + self.right_bins.iter().sum::<u32>()) as f64
                + self.zero_count as f64)
                - 1.0) as f64;
        //println!("quantile limit {}", limit);
        if self.zero_count as f64 > limit {
            return 0.0f64;
        }
        let mut counter = self.zero_count as u32;
        let mut index = 0;
        if self.left_bins.len() > 0 {
            counter += self.left_bins[0];
            while counter as f64 <= limit
                && index < (self.left_margin_mapping.as_ref().unwrap().bins_count - 1)
            {
                index += 1;
                counter += self.left_bins[index];
            }
            if counter as f64 > limit {
                //println!("limit");
                //println!("{}", limit);
                //println!("zero count");
                //println!("{}", self.zero_count);
                //println!("left bins");
                //println!("{:?}", self.left_bins);
                //println!("central bins");
                //println!("{:?}", self.central_bins);
                //println!("right bins");
                //println!("{:?}", self.right_bins);

                return self.left_margin_mapping.as_ref().unwrap().key_to_val_approximation(
                    index as i32 + self.left_margin_mapping.as_ref().unwrap().min_key,
                );
            }
        }
        index = 0;
        counter += self.central_bins[0];
        while counter as f64 <= limit && index < (self.central_mapping.bins_count - 1) {
            index += 1;
            counter += self.central_bins[index];
        }
        if counter as f64 > limit {
            //println!("index {}", index);
            return self.central_mapping.key_to_val_approximation(
                index as i32 + self.central_mapping.min_key,
            );
        }
        index = 0;
        if self.right_bins.len() > 0 {
            counter += self.right_bins[0];
            while counter as f64 <= limit
                && index < (self.right_margin_mapping.as_ref().unwrap().bins_count - 1)
            {
                index += 1;
                counter += self.right_bins[index];
            }
            if counter as f64 > limit {
                return self.right_margin_mapping.as_ref().unwrap().key_to_val_approximation(
                    index as i32 + self.right_margin_mapping.as_ref().unwrap().min_key,
                );
            }
        }
        //we're sure that without right margin bins previous code must return

        //println!("key corresponding to quantile {}", index as i32 + self.mapping.min_key);
        //println!("gamma {}", self.mapping.gamma);
        //println!("min {} max {}", self.mapping.min, self.mapping.max);
        //println!("min key {} max key {}", self.mapping.min_key, self.mapping.max_key);
        //println!("zero count {}", self.zero_count);
        panic!("error estimating quantile");
        self.central_mapping
            .key_to_val_approximation(index as i32 + self.central_mapping.min_key)
    }
}
