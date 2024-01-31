pub mod random;
pub mod simulated_ap;
pub mod simulated_rates;
pub mod five_tuple_aggr;
pub mod constant;
pub mod cached;
pub mod five_tuple;
pub mod oracle;
pub mod onnx_pre_bins;

use std::collections::HashMap;
use std::io::Write;
use crate::packet::{PacketReceiver, Packet};
use std::any::Any;
use std::rc::Rc;
use std::cell::RefCell;
use std::io::{BufRead};
use std::string::String;
use std::vec::Vec;

// dump of all inferences
const DEBUG_INFERENCE: bool = true;
const ONNXRUNTIME_THREAD: usize = 1;
const RANDOM_ORACLE: bool = false;

#[derive(Debug,Clone,Copy)]
pub enum ModelPredictionOutput {
    HeavyHitter,
    Mouse,
}

pub struct ModelWrapperInnerStruct {
    file: String,
    gil: cpython::GILGuard,
    threshold: f64,
    mice_receiver: Option<Rc<RefCell<dyn PacketReceiver>>>,
    hh_receiver: Option<Rc<RefCell<dyn PacketReceiver>>>,
    pub inference_count_hh: usize,
    pub inference_count_mice: usize,
    first_inference: bool,
    desired_ap: f64,
    desired_elephants_misprediction_rate: f64,
    desired_mice_misprediction_rate: f64
}

pub trait ModelWrapper: PacketReceiver {

    fn get_inference_key(&self, packet: &Packet) -> Vec<f64>;

    fn get_inner_struct(&mut self) -> &mut ModelWrapperInnerStruct;

    fn get_total_inference_count(&mut self) -> usize {
        self.get_inner_struct().inference_count_hh + self.get_inner_struct().inference_count_mice
    }

    fn get_hh_inference_count(&mut self) -> usize {
        self.get_inner_struct().inference_count_hh
    }

    fn get_mice_inference_count(&mut self) -> usize {
        self.get_inner_struct().inference_count_mice
    }

    fn set_hh_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>) {
        let reference = self.get_inner_struct();
        reference.hh_receiver = Some(next_block);
    }

    fn set_mice_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>) {
        let reference = self.get_inner_struct();
        reference.mice_receiver = Some(next_block);
    }

    fn configure_model_wrapper(&mut self) {
        let inner_struct = self.get_inner_struct();
        let gil = &mut inner_struct.gil;
        let pickle_file = &inner_struct.file;
        // pickle file path
        //gil.python().run("import pickle", None, None).unwrap();
        //gil.python().run("from sklearn.ensemble import RandomForestClassifier", None, None).unwrap();
        gil.python().run("import numpy as np", None, None).unwrap();
        //gil.python().run("import random", None, None).unwrap();
        //gil.python().run(&format!("f1 = open(\"{}\", \"rb\")", pickle_file), None, None).unwrap();
        //gil.python().run(&format!("classifier: RandomForestClassifier = pickle.load(f1)"), None, None).unwrap();
        // set number of thread
        gil.python().run(&format!("import os"), None, None).unwrap();
        gil.python().run(&format!("import onnxruntime as rt"), None, None).unwrap();
        gil.python().run(&format!("opts = rt.SessionOptions()"), None, None).unwrap();
        gil.python().run(&format!("opts.intra_op_num_threads = 1"), None, None).unwrap();
        gil.python().run(&format!("opts.inter_op_num_threads = 1"), None, None).unwrap();
        gil.python().run(&format!("opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL"), None, None).unwrap();
        gil.python().run(&format!("sess = rt.InferenceSession(\"{}\", sess_options=opts,  providers=[\"CPUExecutionProvider\"])", pickle_file), None, None).unwrap();
        gil.python().run(&format!("input_name = sess.get_inputs()[0].name"), None, None).unwrap();
        gil.python().run(&format!("label_name = sess.get_outputs()[1].name"), None, None).unwrap();
    }

    fn predict_key(&mut self, key: Vec<f64>) -> ModelPredictionOutput {
        let inner_struct = self.get_inner_struct();
        inner_struct.gil.python().run("key = np.array([[]])\n", None, None).unwrap();
        for v in key.to_vec() {
            inner_struct.gil.python().run(&format!("key = np.append(key, {})\n",v), None, None).unwrap();
        }
        //eprintln!("rust key: {:?}", key.to_vec());
        //inner_struct.gil.python().run("print(\"python key: {}\".format(key))", None, None).unwrap();
        //inner_struct.gil.python().run("print(\"python key length: {}\".format(len(key)))", None, None).unwrap();
        //let temp: Vec<Vec<f64>> = inner_struct.gil.python().eval("classifier.predict_proba([key])", None, None).unwrap().extract(inner_struct.gil.python()).unwrap();
        inner_struct.gil.python().run("prediction = sess.run([label_name], {input_name: np.array([key.astype(np.double)])})", None, None).unwrap();
        let prediction: f64 = inner_struct.gil.python().eval("prediction[0][0][1]", None, None).unwrap().extract(inner_struct.gil.python()).unwrap();
        if prediction > inner_struct.threshold {
            self.get_inner_struct().inference_count_hh += 1;
            ModelPredictionOutput::HeavyHitter
        } else {
            self.get_inner_struct().inference_count_mice += 1;
            ModelPredictionOutput::Mouse
        }
    }

    fn predict_key_proba(&mut self, key: Vec<f64>) -> f64 {
        let inner_struct = self.get_inner_struct();
        inner_struct.gil.python().run("key = np.array([[]])\n", None, None).unwrap();
        for v in key.to_vec() {
            inner_struct.gil.python().run(&format!("key = np.append(key, {})\n",v), None, None).unwrap();
        }
        //eprintln!("rust key: {:?}", key.to_vec());
        //inner_struct.gil.python().run("print(\"python key: {}\".format(key))", None, None).unwrap();
        //inner_struct.gil.python().run("print(\"python key length: {}\".format(len(key)))", None, None).unwrap();
        //let temp: Vec<Vec<f64>> = inner_struct.gil.python().eval("classifier.predict_proba([key])", None, None).unwrap().extract(inner_struct.gil.python()).unwrap();
        inner_struct.gil.python().run("prediction = sess.run([label_name], {input_name: np.array([key.astype(np.double)])})", None, None).unwrap();
        let prediction: f64 = inner_struct.gil.python().eval("prediction[0][0][1]", None, None).unwrap().extract(inner_struct.gil.python()).unwrap();
        prediction
    }

    fn dump_debug_inference(&self) -> HashMap<String,Vec<(f64,f64)>>;
}
