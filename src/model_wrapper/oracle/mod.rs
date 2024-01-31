use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::rc::Rc;
use cpython::Python;
use crate::model_wrapper::{DEBUG_INFERENCE, ModelPredictionOutput, ModelWrapper, ModelWrapperInnerStruct};
use crate::packet::{PacketReceiver, Packet, PacketSource};


pub struct ModelWrapperOracle {
    pub inner_struct: ModelWrapperInnerStruct,
    gt_map: HashMap<String, i32>,
    debug_inference_dump: HashMap<String,Vec<(f64,f64)>>,
}

impl ModelWrapperOracle {
    pub fn build(ground_truth_file: String) -> ModelWrapperOracle {
        let inner_struct = ModelWrapperInnerStruct {
            file: ground_truth_file,
            gil: Python::acquire_gil(),
            threshold: 0.0,
            mice_receiver: None,
            hh_receiver: None,
            inference_count_hh: 0,
            inference_count_mice: 0,
            first_inference: true,
            desired_ap: 0.,
            desired_elephants_misprediction_rate: 0.,
            desired_mice_misprediction_rate: 0.,
        };
        let mut ret = ModelWrapperOracle { inner_struct, debug_inference_dump: HashMap::new(), gt_map: HashMap::new()};
        ret.configure_model_wrapper();
        ret
    }
}

impl ModelWrapper for ModelWrapperOracle {
    fn get_inference_key(&self, packet: &Packet) -> Vec<f64> {
        let mut ret = vec![];
        let binary_key = packet.get_binary_key();
        for i in 0..104 {
            ret.push(((binary_key >> (103-i)) & 0x1) as f64);
        }
        ret
    }

    fn get_inner_struct(&mut self) -> &mut ModelWrapperInnerStruct {
        &mut self.inner_struct
    }

    fn configure_model_wrapper(&mut self) {
        let inner_struct = self.get_inner_struct();
        let ground_truth_file = &inner_struct.file;
        let path = Path::new(ground_truth_file);
        let file = File::open(&path).unwrap();
        let reader = BufReader::new(file);

        for line in reader.lines().skip(1) {
            let line = line.unwrap();
            let columns: Vec<&str> = line.split(',').collect();

            let key = columns[0..5].join(",");
            let value: i32 = columns[6].parse().unwrap_or(0);

            self.gt_map.insert(key, value);
        }
    }

    fn predict_key(&mut self, key: Vec<f64>) -> ModelPredictionOutput {
        let mut binary_key: u128 = 0;
        for i in 0..key.len() {
            if key[i] > 0.5 {
                binary_key |= 1 << (103 - i);
            }
        }
        println!("{:?}", binary_key);
        let key_string= Packet::from_binary_key_to_flow_id_string(binary_key);
        println!("{:?}", key_string);
        let class: i32 = *self.gt_map.get(&key_string).unwrap();
        if class == 1 {
            self.get_inner_struct().inference_count_hh += 1;
            ModelPredictionOutput::HeavyHitter
        } else {
            self.get_inner_struct().inference_count_mice += 1;
            ModelPredictionOutput::Mouse
        }
    }

    fn predict_key_proba(&mut self, key: Vec<f64>) -> f64 {
        1.0
    }

    fn dump_debug_inference(&self) -> HashMap<String,Vec<(f64,f64)>> {
        self.debug_inference_dump.clone()
    }
}

impl PacketReceiver for ModelWrapperOracle {
    fn receive_packet<'a>(&'a mut self, mut packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let inference_key = self.get_inference_key(&packet);
        println!("{:?}", inference_key);
        let prediction = self.predict_key(inference_key.to_vec());
        if DEBUG_INFERENCE {
            let key_string = Packet::from_binary_key_to_flow_id_string(packet.get_binary_key());
            let proba =  self.predict_key_proba(inference_key.to_vec());
            packet.agg_features.insert(String::from("prediction_probability"), proba);
            if let Some(v) = self.debug_inference_dump.get_mut(&key_string) {
                v.push((packet.timestamp,proba));
            } else {
                self.debug_inference_dump.insert(key_string, vec![(packet.timestamp,proba)]);
            }
        }
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
    fn as_any(&self) -> &dyn Any {
        self
    }
}
