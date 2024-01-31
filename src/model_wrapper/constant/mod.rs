use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use cpython::Python;
use crate::model_wrapper::{ModelPredictionOutput, ModelWrapper, ModelWrapperInnerStruct};
use crate::packet::{PacketReceiver, Packet, PacketSource};


pub struct ModelWrapperConstant {
    pub inner_struct: ModelWrapperInnerStruct,
    debug_inference_dump: HashMap<String,Vec<(f64,f64)>>,
}
impl ModelWrapperConstant {
    pub fn build() -> ModelWrapperConstant {
        let inner_struct = ModelWrapperInnerStruct {
            file: "".parse().unwrap(),
            gil: Python::acquire_gil(),
            threshold: 0.,
            mice_receiver: None,
            hh_receiver: None,
            inference_count_hh: 0,
            inference_count_mice: 0,
            first_inference: true,
            desired_ap: 0.,
            desired_elephants_misprediction_rate: 0.,
            desired_mice_misprediction_rate: 0.
        };
        let mut ret = ModelWrapperConstant { inner_struct, debug_inference_dump: HashMap::new(), };
        ret.configure_model_wrapper();
        ret
    }
}


impl ModelWrapper for ModelWrapperConstant {

    fn get_inference_key(&self, packet: &Packet) -> Vec<f64> {
        let mut ret = vec![];
        let binary_key = packet.get_binary_key();
        for i in 0..96{
            ret.push(((binary_key >> (103-i)) & 0x1) as f64);
        }
        ret.push(1.0);
        ret.push(*packet.agg_features.get("avg_size").unwrap() as f64);
        ret.push(*packet.agg_features.get("stdev_size").unwrap() as f64);
        ret.push(*packet.agg_features.get("avg_iat").unwrap() as f64);
        ret.push(*packet.agg_features.get("stdev_iat").unwrap() as f64);
        ret
    }

    fn get_inner_struct(&mut self) -> &mut ModelWrapperInnerStruct {
        &mut self.inner_struct
    }

    fn dump_debug_inference(&self) -> HashMap<String,Vec<(f64,f64)>> {
        self.debug_inference_dump.clone()
    }
}

impl PacketReceiver for ModelWrapperConstant {
    fn receive_packet<'a>(&'a mut self, mut packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        let inference_key = self.get_inference_key(&packet);
        let inner_struct = self.get_inner_struct();
        let prediction = ModelPredictionOutput::Mouse;
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
