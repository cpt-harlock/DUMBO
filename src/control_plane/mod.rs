use std::{rc::Rc, cell::RefCell};
use std::collections::HashSet;
use crate::packet::{PacketReceiver, Packet, PacketSource};

pub struct ControlPlane {
    cms_keys: HashSet<u128>,
    cms_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
    cms_debug_data: Vec<(f64,String)>,
    hash_table_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
    cms_keys_accesses: usize,
}

impl PacketReceiver for ControlPlane {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
       self.cms_keys_accesses += 1;
       self.cms_keys.insert(packet.get_binary_key());
       self.cms_debug_data.push((packet.timestamp, Packet::from_binary_key_to_flow_id_string(packet.get_binary_key())));
       let ret = None;
       return ret;
    }
}

impl ControlPlane {
    pub fn build_control_plane() -> ControlPlane {
        ControlPlane { cms_keys: HashSet::new(), cms_block: None, cms_debug_data: vec![], hash_table_block: None, cms_keys_accesses: 0 }
    }

    pub fn get_transmitted_keys_count(&self) -> usize {
        self.cms_keys_accesses
    }

    pub fn get_debug_data(&self) -> Vec<(f64,String)> {
        self.cms_debug_data.to_vec()
    }

    pub fn get_keys(&self) -> HashSet<u128> {
        self.cms_keys.clone()
    }

    pub fn clear(&mut self) {
        self.cms_keys.clear();
    }
}
