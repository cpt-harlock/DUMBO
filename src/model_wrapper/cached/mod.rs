use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::rc::Rc;
use cpython::Python;
use crate::model_wrapper::{RANDOM_ORACLE, DEBUG_INFERENCE, ModelPredictionOutput, ModelWrapper, ModelWrapperInnerStruct};
use crate::packet::{PacketReceiver, Packet, PacketSource};


pub struct ModelWrapperCached {
    pub inner_struct: ModelWrapperInnerStruct,
    inference_cache: HashMap<u128,f64>,
    debug_inference_dump: HashMap<String,Vec<(f64,f64)>>,
}

impl ModelWrapperCached {
    pub fn build(inference_cache_file: String, threshold: f64) -> ModelWrapperCached {
        let inner_struct = ModelWrapperInnerStruct {
            file: "".parse().unwrap(),
            gil: Python::acquire_gil(),
            threshold,
            mice_receiver: None,
            hh_receiver: None,
            inference_count_hh: 0,
            inference_count_mice: 0,
            first_inference: true,
            desired_ap: 0.,
            desired_elephants_misprediction_rate: 0.,
            desired_mice_misprediction_rate: 0.
        };
        let mut inference_cache = HashMap::<_,_>::new();
        let file = File::open(inference_cache_file).unwrap();
        let read_buff = BufReader::new(file);
        for line in read_buff.lines() {
            let rec = line.unwrap().split(",").map(|v| {String::from(v)}).collect::<Vec<String>>();
            // il solito bagno de sangue...
            let src_ip_rec = rec[0].split(".").map(|v| {String::from(v)}).into_iter().map(|v| {v.parse::<u8>().unwrap()}).collect::<Vec<u8>>();
            let dst_ip_rec = rec[1].split(".").map(|v| {String::from(v)}).into_iter().map(|v| {v.parse::<u8>().unwrap()}).collect::<Vec<u8>>();
            let src_port = rec[2].parse::<usize>().unwrap();
            let dst_port = rec[3].parse::<usize>().unwrap();
            let proto = rec[4].parse::<usize>().unwrap();
            let proba = rec[5].parse::<f64>().unwrap();
            // build key
            let mut key: u128 = src_ip_rec[0] as u128;
            for i in 1..4 {
                key = (key << 8) | (src_ip_rec[i] as u128);
            }
            for i in 0..4 {
                key = (key << 8) | (dst_ip_rec[i] as u128);
            }
            key = (key << 16) | (src_port as u128);
            key = (key << 16) | (dst_port as u128);
            key = (key << 8) | (proto as u128);
            inference_cache.insert(key, proba);
        }
        let mut ret = ModelWrapperCached { inner_struct, inference_cache, debug_inference_dump: HashMap::new(), };
        ret.configure_model_wrapper();
        ret
    }

    pub fn dump_debug_inference(&self) -> HashMap<String,Vec<(f64,f64)>> {
        self.debug_inference_dump.clone()
    }
}

impl ModelWrapper for ModelWrapperCached {

    fn get_inference_key(&self, packet: &Packet) -> Vec<f64> {
        let mut ret = vec![];
        let binary_key = packet.get_binary_key();
        for i in 0..96{
            ret.push(((binary_key >> (103-i)) & 0x1) as f64);
        }
        if !RANDOM_ORACLE {
            ret.push(1.0);
        }
        ret
    }

    fn get_inner_struct(&mut self) -> &mut ModelWrapperInnerStruct {
        &mut self.inner_struct
    }

    fn dump_debug_inference(&self) -> HashMap<String,Vec<(f64,f64)>> {
        self.debug_inference_dump.clone()
    }
}

impl PacketReceiver for ModelWrapperCached {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn receive_packet<'a>(&'a mut self, mut packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {

        let inference_key = self.get_inference_key(&packet);
        let prediction = if let Some(inference) = self.inference_cache.get(&(packet.get_binary_key())) {
            packet.agg_features.insert(String::from("prediction_probability"), *inference);
            if DEBUG_INFERENCE {
                let key_string =  Packet::from_binary_key_to_flow_id_string(packet.get_binary_key());
                if let Some(v) = self.debug_inference_dump.get_mut(&key_string) {
                    v.push((packet.timestamp, *inference));
                } else {
                    self.debug_inference_dump.insert(key_string, vec![(packet.timestamp, *inference)]);
                }
            }
            if *inference > self.inner_struct.threshold {
                ModelPredictionOutput::HeavyHitter
            } else {
                ModelPredictionOutput::Mouse
            }
        } else {
            let proba = self.predict_key_proba(inference_key.to_vec());
            packet.agg_features.insert(String::from("prediction_probability"), proba);
            if DEBUG_INFERENCE {
                let key_string =  Packet::from_binary_key_to_flow_id_string(packet.get_binary_key());
                if let Some(v) = self.debug_inference_dump.get_mut(&key_string) {
                    v.push((packet.timestamp, proba));
                } else {
                    self.debug_inference_dump.insert(key_string, vec![(packet.timestamp, proba)]);
                }
            }
            let temp = if proba > self.inner_struct.threshold {
                ModelPredictionOutput::HeavyHitter
            } else {
                ModelPredictionOutput::Mouse
            };
            self.inference_cache.insert(packet.get_binary_key(),proba);
            temp
        };
        let inner_struct = self.get_inner_struct();
        packet.packet_source = PacketSource::FromModel;

        //eprintln!("prediction: {:?}", prediction);
        match prediction {
            ModelPredictionOutput::HeavyHitter => {
                packet.agg_features.insert(String::from("label"), 1.0);
                return inner_struct.hh_receiver.clone();
            },
            ModelPredictionOutput::Mouse => {
                packet.agg_features.insert(String::from("label"), 0.0);
                return inner_struct.mice_receiver.clone();
            },
        }
    }
}