use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use cpython::Python;
use crate::model_wrapper::{RANDOM_ORACLE, DEBUG_INFERENCE, ModelPredictionOutput, ModelWrapper, ModelWrapperInnerStruct};
use crate::packet::{PacketReceiver, Packet, PacketSource};


pub struct ModelWrapperFiveTuple {
    pub inner_struct: ModelWrapperInnerStruct,
    debug_inference_dump: HashMap<String,Vec<(f64,f64)>>,
    features_file: String,
}

impl ModelWrapperFiveTuple {
    pub fn build(pickle_file: String, threshold: f64, features_file: String) -> ModelWrapperFiveTuple {
        let inner_struct = ModelWrapperInnerStruct {
            file: pickle_file,
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
        let mut ret = ModelWrapperFiveTuple { inner_struct, debug_inference_dump: HashMap::new(), features_file: features_file.clone() };
        ret.configure_model_wrapper();
        // create once vector with features name used later to create the panda dataframe
        let gil = &mut ret.get_inner_struct().gil;
        gil.python().run(&format!("import pandas as pd\n"), None, None).unwrap();
        gil.python().run(&format!("import pickle\n"), None, None).unwrap();
        gil.python().run(&format!("features_name_vector = []\n"), None, None).unwrap();
        for i in 0..32 {
            gil.python().run(&format!("features_name_vector.append(\"src_ip_{}\")\n",i), None, None).unwrap();
        }
        for i in 0..32 {
            gil.python().run(&format!("features_name_vector.append(\"dst_ip_{}\")\n",i), None, None).unwrap();
        }
        for i in 0..16 {
            gil.python().run(&format!("features_name_vector.append(\"src_port_{}\")\n",i), None, None).unwrap();
        }
        for i in 0..16 {
            gil.python().run(&format!("features_name_vector.append(\"dst_port_{}\")\n",i), None, None).unwrap();
        }
        gil.python().run(&format!("features_name_vector.append(\"protocol\")\n"), None, None).unwrap();
        //import the used features vector
        gil.python().run(&format!("features_file = open(\"{}\", \"rb\")\n", features_file), None, None).unwrap();
        gil.python().run(&format!("used_features_vector = pickle.load(features_file)\n"), None, None).unwrap();
        ret
    }

    pub fn dump_debug_inference(&self) -> HashMap<String,Vec<(f64,f64)>> {
        self.debug_inference_dump.clone()
    }
}


impl ModelWrapper for ModelWrapperFiveTuple {

    fn get_inference_key(&self, packet: &Packet) -> Vec<f64> {
        let mut ret = vec![];
        let binary_key = packet.get_binary_key();
        for i in 0..96 {
            ret.push(((binary_key >> (103-i)) & 0x1) as f64);
        }
        //TODO: this should be deleted once we remove from normal oracles the TCP bit
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


impl PacketReceiver for ModelWrapperFiveTuple {
    fn receive_packet<'a>(&'a mut self, mut packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let inference_key = self.get_inference_key(&packet);
        //create inference key and transform in a panda dataframe
        let gil = &mut self.get_inner_struct().gil;
        gil.python().run("key = np.array([])\n", None, None).unwrap();
        for v in inference_key.to_vec() {
            gil.python().run(&format!("key = np.append(key, {:?})\n",v), None, None).unwrap();
        }
        gil.python().run("X = pd.DataFrame(data=key.reshape((1,) + key.shape), columns=features_name_vector)\n", None, None).unwrap();
        gil.python().run("X_onnx = X[used_features_vector].astype(np.float32)\n", None, None).unwrap();
        gil.python().run("inputs = { c: X_onnx[c].values.reshape((-1, 1)) for c  in X_onnx.columns }\n", None, None).unwrap();
        gil.python().run("onnx_preds = sess.run(None, inputs)[1][:,1][0]\n", None, None).unwrap();
        let prediction_prob: f64= gil.python().eval("onnx_preds\n", None, None).unwrap().extract(gil.python()).unwrap();

        if DEBUG_INFERENCE {
            let key_string = Packet::from_binary_key_to_flow_id_string(packet.get_binary_key());
            packet.agg_features.insert(String::from("prediction_probability"), prediction_prob);
            if let Some(v) = self.debug_inference_dump.get_mut(&key_string) {
                v.push((packet.timestamp, prediction_prob));
            } else {
                self.debug_inference_dump.insert(key_string, vec![(packet.timestamp,prediction_prob)]);
            }
        }
        let inner_struct = self.get_inner_struct();
        packet.packet_source = PacketSource::FromModel;
        //eprintln!("prediction: {:?}", prediction);
        let prediction = if prediction_prob >= inner_struct.threshold {
            ModelPredictionOutput::HeavyHitter
        } else {
            ModelPredictionOutput::Mouse
        };
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
