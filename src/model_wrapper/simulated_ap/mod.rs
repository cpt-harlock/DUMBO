use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::rc::Rc;
use cpython::Python;
use crate::model_wrapper::{DEBUG_INFERENCE, ModelPredictionOutput, ModelWrapper, ModelWrapperInnerStruct};
use crate::packet::{PacketReceiver, Packet, PacketSource};


pub struct ModelWrapperSimulatedAP {
    pub inner_struct: ModelWrapperInnerStruct,
    gt_map: HashMap<String, i32>,
    debug_inference_dump: HashMap<String,Vec<(f64,f64)>>,
}

impl ModelWrapperSimulatedAP {
    pub fn build(ground_truth_file: String, desired_ap: f64, threshold: f64) -> ModelWrapperSimulatedAP {
        let inner_struct = ModelWrapperInnerStruct {
            file: ground_truth_file,
            gil: Python::acquire_gil(),
            threshold,
            mice_receiver: None,
            hh_receiver: None,
            inference_count_hh: 0,
            inference_count_mice: 0,
            first_inference: true,
            desired_ap,
            desired_elephants_misprediction_rate: 0.,
            desired_mice_misprediction_rate: 0.,
        };
        let mut ret = ModelWrapperSimulatedAP { inner_struct, debug_inference_dump: HashMap::new(), gt_map: HashMap::new()};
        ret.configure_model_wrapper();
        ret
    }
}

impl ModelWrapper for ModelWrapperSimulatedAP {
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
        let desired_ap = &inner_struct.desired_ap;
        let gil = &mut inner_struct.gil;
        let path = Path::new(ground_truth_file);
        let alpha_to_ap_file = path.with_extension("json");

        gil.python().run(&format!("import numpy as np"), None, None).unwrap();
        gil.python().run(&format!("import pandas as pd"), None, None).unwrap();
        gil.python().run(&format!("import json"), None, None).unwrap();
        gil.python().run(&format!("import hashlib"), None, None).unwrap();
        gil.python().run(&format!("df = pd.read_csv(\"{}\")", ground_truth_file), None, None).unwrap();
        gil.python().run(&format!("df['key'] = df['ip_src'] + ',' + df['ip_dst'] + ',' + df['port_src'].astype(str) + ',' + df['port_dst'].astype(str) + ',' + df['proto'].astype(str)"), None, None).unwrap();
        gil.python().run(&format!("dataset = df.set_index('key')['heavy_hitter'].to_dict()"), None, None).unwrap();
        gil.python().run(&format!("alpha_to_ap_file = \"{}\".replace('.csv', '.json')", ground_truth_file), None, None).unwrap();
        gil.python().run(&format!("simulate_predict = lambda key, alpha: 1 - np.random.beta(alpha, 3) if dataset[key] == 1 else np.random.beta(alpha, 3)"), None, None).unwrap();
        if fs::metadata(alpha_to_ap_file).is_ok() {
            gil.python().run(&format!("alpha_to_ap = json.load(open(alpha_to_ap_file, 'r'))"), None, None).unwrap();
        } else {
            gil.python().run(&format!("from sklearn.metrics import average_precision_score"), None, None).unwrap();
            gil.python().run(&format!("y_true = np.array([v for v in dataset.values()])"), None, None).unwrap();
            gil.python().run(&format!("alpha_to_ap = {{ 3*(n/135/3): average_precision_score(y_true, np.array([simulate_predict(k, 3*(n/135/3)) for k in dataset.keys()])) for n in range(1, 100*3) }}"), None, None).unwrap();
            gil.python().run(&format!("json.dump(alpha_to_ap, open(alpha_to_ap_file, 'w'))"), None, None).unwrap();
        }
        gil.python().run(&format!("alpha = np.mean([float(a) for a, AP in alpha_to_ap.items() if float(f'{{AP:.2}}') == {}])", desired_ap), None, None).unwrap();
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

        gil.python().run(&format!("s = int.from_bytes(hashlib.sha256(str({}).encode()).digest()[:4], 'big')", binary_key), None, None).unwrap();
        gil.python().run(&format!("np.random.seed(s)"), None, None).unwrap();
        gil.python().run(&format!("prediction = simulate_predict(\"{}\", alpha)", key_string), None, None).unwrap();

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
        let mut binary_key: u128 = 0;
        for i in 0..key.len() {
            if key[i] > 0.5 {
                binary_key |= 1 << (103 - i);
            }
        }
        let key_string= Packet::from_binary_key_to_flow_id_string(binary_key);
        let inner_struct = self.get_inner_struct();
        let gil = &mut inner_struct.gil;

        gil.python().run(&format!("np.random.seed({})", binary_key as u32), None, None).unwrap();
        gil.python().run(&format!("prediction = simulate_predict(\"{}\", alpha)", key_string), None, None).unwrap();

        let prediction: f64 = inner_struct.gil.python().eval("prediction", None, None).unwrap().extract(inner_struct.gil.python()).unwrap();
        prediction
    }

    fn dump_debug_inference(&self) -> HashMap<String,Vec<(f64,f64)>> {
        self.debug_inference_dump.clone()
    }
}

impl PacketReceiver for ModelWrapperSimulatedAP {
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
