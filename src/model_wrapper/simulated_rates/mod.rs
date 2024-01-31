use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::rc::Rc;
use cpython::Python;
use crate::model_wrapper::{DEBUG_INFERENCE, ModelPredictionOutput, ModelWrapper, ModelWrapperInnerStruct};
use crate::packet::{PacketReceiver, Packet, PacketSource};


pub struct ModelWrapperSimulatedRates {
    pub inner_struct: ModelWrapperInnerStruct,
    gt_map: HashMap<String, i32>,
    debug_inference_dump: HashMap<String,Vec<(f64,f64)>>,
}

impl ModelWrapperSimulatedRates {
    pub fn build(ground_truth_file: String, elephants_misprediction_rate: f64, mice_misprediction_rate: f64) -> ModelWrapperSimulatedRates {
        let inner_struct = ModelWrapperInnerStruct {
            file: ground_truth_file,
            gil: Python::acquire_gil(),
            threshold: 0.,
            mice_receiver: None,
            hh_receiver: None,
            inference_count_hh: 0,
            inference_count_mice: 0,
            first_inference: true,
            desired_ap: 0.,
            desired_elephants_misprediction_rate: elephants_misprediction_rate,
            desired_mice_misprediction_rate: mice_misprediction_rate,
        };
        let mut ret = ModelWrapperSimulatedRates { inner_struct, debug_inference_dump: HashMap::new(), gt_map: HashMap::new() };
        ret.configure_model_wrapper();
        ret
    }
}

impl ModelWrapper for ModelWrapperSimulatedRates {
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
        let gil = &mut inner_struct.gil;
        let path = Path::new(ground_truth_file);

        gil.python().run(&format!("import numpy as np"), None, None).unwrap();
        gil.python().run(&format!("import pandas as pd"), None, None).unwrap();
        gil.python().run(&format!("import hashlib"), None, None).unwrap();
        gil.python().run(&format!("df = pd.read_csv(\"{}\")", ground_truth_file), None, None).unwrap();
        gil.python().run(&format!("df['key'] = df['ip_src'] + ',' + df['ip_dst'] + ',' + df['port_src'].astype(str) + ',' + df['port_dst'].astype(str) + ',' + df['proto'].astype(str)"), None, None).unwrap();
        gil.python().run(&format!("dataset = df.set_index('key')['heavy_hitter'].to_dict()"), None, None).unwrap();
        gil.python().run(&format!("simulate_predict = lambda key, pmr, nmr: int(np.random.rand() > pmr) if dataset[key] == 1 else int(np.random.rand() < nmr)"), None, None).unwrap();
    }

    fn predict_key(&mut self, key: Vec<f64>) -> ModelPredictionOutput {
        let mut binary_key: u128 = 0;
        for i in 0..key.len() {
            if key[i] > 0.5 {
                binary_key |= 1 << (103 - i);
            }
        }
        let key_string= Packet::from_binary_key_to_flow_id_string(binary_key);
        let inner_struct = self.get_inner_struct();
        let gil = &mut inner_struct.gil;
        let pmr = inner_struct.desired_elephants_misprediction_rate;
        let nmr = inner_struct.desired_mice_misprediction_rate;

        gil.python().run(&format!("s = int.from_bytes(hashlib.sha256(str({}).encode()).digest()[:4], 'big')", binary_key), None, None).unwrap();
        gil.python().run(&format!("np.random.seed(s)"), None, None).unwrap();
        gil.python().run(&format!("prediction = simulate_predict(\"{}\", {}, {})", key_string, pmr, nmr), None, None).unwrap();

        let prediction: f64 = inner_struct.gil.python().eval("prediction", None, None).unwrap().extract(inner_struct.gil.python()).unwrap();

        if prediction > inner_struct.threshold {
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

impl PacketReceiver for ModelWrapperSimulatedRates {
    fn receive_packet<'a>(&'a mut self, mut packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let inference_key = self.get_inference_key(&packet);
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
