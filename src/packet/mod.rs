use std::collections::HashMap;
use std::any::Any;
use std::rc::Rc;
use std::cell::RefCell;
use std::time::Instant;

#[derive(Clone,Debug)]
pub enum PacketSource {
    FromHashTable,
    FromHashTableDiscarded,
    FromParser,
    FromModel,
    FromCms,
    FromBf,
    FromFlowManager,
    Other,
}

#[derive(Clone)]
pub struct Packet {
    pub srcIp: u32,
    pub dstIp: u32,
    pub srcPort: u16,
    pub dstPort: u16,
    pub protoType: u8,
    pub timestamp: f64,
    pub agg_features: HashMap<String,f64>,
    pub packet_source: PacketSource,
    // use to insert more than one packet
    pub pkt_count: usize,
    pub has_fin_rst: bool,
}

pub trait PacketReceiver {
    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>>;
    fn as_any(&self) -> &dyn Any;
}


pub struct PacketSink {
}

impl PacketReceiver for PacketSink {
    fn receive_packet<'a>(&'a mut self, packet: &'a mut Packet) -> Option<Rc<RefCell<dyn PacketReceiver>>> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Packet {

    pub fn clone_from_existing_packet(&mut self, packet: &Packet) {
        self.srcIp = packet.srcIp;
        self.dstIp = packet.dstIp;
        self.srcPort = packet.srcPort;
        self.dstPort = packet.dstPort;
        self.protoType = packet.protoType;
        self.timestamp = packet.timestamp;
        self.agg_features = packet.agg_features.clone();
        self.packet_source = packet.packet_source.clone();
        self.pkt_count = packet.pkt_count;
    }

    pub fn build_packet(srcIp: u32, dstIp: u32, srcPort: u16, dstPort: u16, protoType: u8) -> Packet {
        let agg_features = HashMap::new();
        Packet {
            srcIp,
            dstIp,
            srcPort,
            dstPort,
            protoType,
            agg_features,
            timestamp: 0.0,
            packet_source: PacketSource::Other,
            pkt_count: 1,
            has_fin_rst: false,
        }
    }

    pub fn build_packet_timestamp(srcIp: u32, dstIp: u32, srcPort: u16, dstPort: u16, protoType: u8, timestamp: f64) -> Packet {
        let agg_features = HashMap::new();
        Packet {
            srcIp,
            dstIp,
            srcPort,
            dstPort,
            protoType,
            agg_features,
            timestamp,
            packet_source: PacketSource::Other,
            pkt_count: 1,
            has_fin_rst: false,
        }
    }

    pub fn build_packet_timestamp_size(srcIp: u32, dstIp: u32, srcPort: u16, dstPort: u16, protoType: u8, timestamp: f64, size: usize) -> Packet {
        let mut agg_features = HashMap::new();
        agg_features.insert(String::from("Size"), size as f64);
        Packet {
            srcIp,
            dstIp,
            srcPort,
            dstPort,
            protoType,
            agg_features,
            timestamp,
            packet_source: PacketSource::Other,
            pkt_count: 1,
            has_fin_rst: false,
        }
    }

    pub fn build_from_binary_key(binary_key: u128) -> Packet {
        let srcIp = ((binary_key >> 72) & 0xFFFF_FFFF) as u32;
        let dstIp = ((binary_key >> 40) & 0xFFFF_FFFF) as u32;
        let srcPort = ((binary_key >> 24) & 0xFFFF) as u16;
        let dstPort = ((binary_key >> 8) & 0xFFFF) as u16;
        let protoType = ((binary_key) & 0xFF) as u8;
        let agg_features = HashMap::new();

        Packet {
            srcIp,
            dstIp,
            srcPort,
            dstPort,
            protoType,
            agg_features,
            timestamp : 0.0,
            packet_source: PacketSource::Other,
            pkt_count: 1,
            has_fin_rst: false,
        }
    }

    pub fn from_binary_key_to_flow_id_string(binary_key: u128) -> String {
        let srcIp = ((binary_key >> 72) & 0xFFFF_FFFF) as u32;
        let dstIp = ((binary_key >> 40) & 0xFFFF_FFFF) as u32;
        let srcPort = ((binary_key >> 24) & 0xFFFF) as u16;
        let dstPort = ((binary_key >> 8) & 0xFFFF) as u16;
        let protoType = ((binary_key) & 0xFF) as u8;
        let srcIp_vec = srcIp.to_be_bytes();
        let dstIp_vec = dstIp.to_be_bytes();
        let ret = format!("{}.{}.{}.{},{}.{}.{}.{},{},{},{}", srcIp_vec[0], srcIp_vec[1], srcIp_vec[2], srcIp_vec[3], dstIp_vec[0], dstIp_vec[1], dstIp_vec[2], dstIp_vec[3], srcPort, dstPort, protoType);
        ret
    }

    pub fn build_from_binary_key_timestamp(binary_key: u128, timestamp: f64) -> Packet {
        let srcIp = ((binary_key >> 72) & 0xFFFF_FFFF) as u32;
        let dstIp = ((binary_key >> 40) & 0xFFFF_FFFF) as u32;
        let srcPort = ((binary_key >> 24) & 0xFFFF) as u16;
        let dstPort = ((binary_key >> 8) & 0xFFFF) as u16;
        let protoType = ((binary_key) & 0xFF) as u8;
        let agg_features = HashMap::new();

        Packet {
            srcIp,
            dstIp,
            srcPort,
            dstPort,
            protoType,
            agg_features,
            timestamp,
            packet_source: PacketSource::Other,
            pkt_count: 1,
            has_fin_rst: false,
        }
    }

    pub fn get_binary_key(&self) -> u128 {
        let mut ret: u128 = 0;
        ret = self.srcIp as u128;
        ret = (ret << 32) | (self.dstIp as u128);
        ret = (ret << 16) | (self.srcPort as u128);
        ret = (ret << 16) | (self.dstPort as u128);
        ret = (ret << 8) | (self.protoType as u128);
        ret
    }
}
