use super::packet::{PacketReceiver, PacketSource};
pub use pcap_parser::traits::PcapReaderIterator;
pub use pcap_parser::*;
use pcap_parser::data::{get_packetdata, PacketData};
use std::collections::HashMap;
pub use std::fs::File;
pub use std::io::ErrorKind;
pub use std::io::Read;
pub use packet::ether::Packet as EthernetPacket;
pub use packet::ip::Packet as IpPacket;
pub use packet::tcp::Packet as TcpPacket;
pub use packet::tcp::Flags as TcpFlags;
pub use packet::udp::Packet as UdpPacket;
pub use packet::Packet;
pub use csv::Writer;
pub use std::io::Write;
use std::cell::RefCell;
use std::rc::Rc;
use packet::tcp::Flags;
use crate::pcap::IcmpPacket;


enum PacketType {
    Tcp,
    Udp,
    Icmp,
}

pub struct Parser {
    pcap_file: String,
    tcp_only: bool,
    pub next_block: Option<Rc<RefCell<dyn PacketReceiver>>>,
    packet_list: Vec<crate::packet::Packet>,
    statistics: HashMap<u128,HashMap<String,Vec<f64>>>,
}

impl Parser {
    pub fn build_parser(pcap_file: &str, tcp_only: bool) -> Parser {
        let statistics = HashMap::new();
        let pp = Parser {
            pcap_file: String::from(pcap_file),
            tcp_only,
            next_block: None,
            packet_list: vec![],
            statistics,
        };
        pp
    }

    pub fn set_next_block(&mut self, next_block: Rc<RefCell<dyn PacketReceiver>>) {
        self.next_block = Some(next_block);
    }

    pub fn build_parser_next_block(pcap_file: &str, tcp_only: bool, next_block: Rc<RefCell<dyn PacketReceiver>>) -> Parser {
        let statistics = HashMap::new();
        let pp = Parser {
            pcap_file: String::from(pcap_file),
            tcp_only,
            next_block: Some(next_block),
            packet_list: vec![],
            statistics,
        };
        pp
    }

    fn update_statistics(&mut self, packet: &crate::packet::Packet) {
        // for the moment, save flow size, timestamps and  interarrival time
        let flow_key = packet.get_binary_key();
        let timestamp = packet.timestamp;
        if let Some(f) = self.statistics.get_mut(&flow_key) {
            // size update
            *(f.get_mut(&String::from("size")).unwrap().get_mut(0).unwrap()) += 1.0;
            let last_timestamp = *(f.get_mut(&String::from("timestamps")).unwrap().last().unwrap());
            if let Some(iats) = f.get_mut("iats") {
                iats.push(timestamp - last_timestamp);
            } else {
                f.insert(String::from("iats"), vec![timestamp - last_timestamp]);
            }
            f.get_mut(&String::from("timestamps")).unwrap().push(timestamp);
        } else {
            let mut temp = HashMap::new();
            temp.insert(String::from("size"), vec![1.0]);
            temp.insert(String::from("timestamps"), vec![timestamp]);
            self.statistics.insert(flow_key, temp);
        }
    }

    pub fn dump_statistics_formatted(&self) -> Vec<String> {
        let mut ret = vec![];
        for (k,h) in &self.statistics {
            let flow_id = crate::packet::Packet::from_binary_key_to_flow_id_string(*k);
            let size = *h.get("size").unwrap().get(0).unwrap() as usize;
            let string = format!("{},{}", flow_id, size);
            ret.push(string);
        }
        ret
    }

    pub fn dump_statistics(&self) -> HashMap<u128,usize> {
        let mut ret = HashMap::new();
        for (k,h) in &self.statistics {
            let size = *h.get("size").unwrap().get(0).unwrap() as usize;
            ret.insert(*k,size);
        }
        ret
    }

    fn octets_to_ip(octects: [u8; 4]) -> u32 {
        let mut ret;
        ret = octects[0] as u32;
        ret = (ret << 8) | (octects[1] as u32);
        ret = (ret << 8) | (octects[2] as u32);
        ret = (ret << 8) | (octects[3] as u32);
        ret
    }

    /**
     * Populate packet list
     */

    pub fn run(&mut self) -> (Option<crate::packet::Packet>, Option<Rc<RefCell<dyn PacketReceiver>>>) {
        if let Some(_) = self.packet_list.first() {
            let ret_pkt = self.packet_list.remove(0);
            (Some(ret_pkt), self.next_block.clone())
        } else {
            (None, None)
        }
    }

    pub fn run_all(&mut self) -> Vec<(Option<crate::packet::Packet>, Option<Rc<RefCell<dyn PacketReceiver>>>)> {
        let mut vec = vec![];
        for p in &self.packet_list {
            vec.push((Some(p.clone()), self.next_block.clone()));
        }
        vec.push((None, None));
        self.packet_list.clear();
        vec
    }

    pub fn get_packet_list(&self) -> Vec<crate::packet::Packet> {
        self.packet_list.to_vec()
    }


    pub fn parse_pcap(&mut self) {
        if let None = self.next_block {
            println!("No next block in simulation pipeline, connect before running");
            return;
        }
        let mut if_linktypes = Vec::new();
        let mut trace_linktype;
        let mut file = File::open(&self.pcap_file).unwrap();
        let mut buffer = Vec::new();

        file.read_to_end(&mut buffer).unwrap();
        let mut num_packets = 0;
        // try pcap first
        match PcapCapture::from_file(&buffer) {
            Ok(capture) => {
                println!("Format: PCAP");
                //settin PCAP packet type
                trace_linktype = capture.header.network;
                for block in capture.iter() {
                    match block {
                        PcapBlock::NG(Block::SectionHeader(ref _shb)) => {
                            // starting a new section, clear known interfaces
                            if_linktypes = Vec::new();
                            println!("ng block header");
                        }
                        PcapBlock::NG(Block::InterfaceDescription(ref idb)) => {
                            if_linktypes.push(idb.linktype);
                            println!("ng block interface desc");
                        }
                        PcapBlock::NG(Block::EnhancedPacket(ref epb)) => {
                            assert!((epb.if_id as usize) < if_linktypes.len());
                            let linktype = if_linktypes[epb.if_id as usize];
                            println!("ng block enh pack");
                            #[cfg(feature = "data")]
                            let res = pcap_parser::data::get_packetdata(
                                epb.data,
                                linktype,
                                epb.caplen as usize,
                            );
                        }
                        PcapBlock::NG(Block::SimplePacket(ref spb)) => {
                            assert!(if_linktypes.len() > 0);
                            println!("ng block simple pack");
                            let linktype = if_linktypes[0];
                            let blen = (spb.block_len1 - 16) as usize;
                            #[cfg(feature = "data")]
                            let res = pcap_parser::data::get_packetdata(spb.data, linktype, blen);
                        }
                        PcapBlock::NG(_) => {
                            // can be statistics (ISB), name resolution (NRB), etc.
                            println!("ng block unsup");
                            eprintln!("unsupported block");
                        }
                        PcapBlock::Legacy(packet) => {
                            let pkt_data = get_packetdata(packet.data, trace_linktype, packet.caplen as usize).unwrap();
                            //println!("usec {}",packet.ts_sec as f64 + packet.ts_usec as f64 / 1000000.0);
                            let ts = packet.ts_sec as f64 + (packet.ts_usec as f64 /1000000.0);
                            let l3_packet;
                            let mut l4_packet : TcpPacket<&[u8]> = TcpPacket::unchecked(&[0]);
                            let mut l4_packet_udp : UdpPacket<&[u8]>= UdpPacket::unchecked(&[0]);
                            let mut l4_packet_icmp : IcmpPacket<&[u8]> = IcmpPacket::unchecked(&[0]);
                            let l2_packet;
                            let packet_type;
                            let mut is_last = false;

                            match pkt_data {
                                PacketData::L2(a) => {
                                    l2_packet = EthernetPacket::new(a).unwrap();
                                    //unchecked as there's no payload
                                    if l2_packet.protocol() != packet::ether::Protocol::Ipv4 {
                                        continue;
                                    }
                                    let temp_l3 = IpPacket::unchecked(l2_packet.payload());
                                    match temp_l3 {
                                        IpPacket::V4(p) => {
                                            l3_packet = p;
                                        },
                                        _ => { continue; }
                                    }
                                    if l3_packet.protocol() == packet::ip::Protocol::Tcp {
                                        l4_packet = TcpPacket::unchecked(l3_packet.payload());
                                        packet_type = PacketType::Tcp;
                                        is_last = l4_packet.flags().contains(Flags::FIN) || l4_packet.flags().contains(Flags::RST);
                                    } else if l3_packet.protocol() == packet::ip::Protocol::Udp {
                                        if self.tcp_only { continue; }
                                        if l3_packet.payload().len() < 8 { continue; }
                                        l4_packet_udp = UdpPacket::no_payload(l3_packet.payload()).unwrap();
                                        packet_type = PacketType::Udp;
                                    } else if l3_packet.protocol() == packet::ip::Protocol::Icmp {
                                        continue;
                                        if self.tcp_only { continue; }
                                        l4_packet_icmp = IcmpPacket::unchecked(l3_packet.payload());
                                        packet_type = PacketType::Icmp;
                                    } else {
                                        continue;
                                    }
                                },
                                PacketData::L3(_, b) => {
                                    let temp_l3 = IpPacket::unchecked(b);
                                    match temp_l3 {
                                        IpPacket::V4(p) => {l3_packet = p; },
                                        _ => { continue; }
                                    }
                                    if l3_packet.protocol() == packet::ip::Protocol::Tcp {
                                        match TcpPacket::new(l3_packet.payload()) {
                                            Ok(p) => {
                                                l4_packet = p;
                                                is_last = l4_packet.flags().contains(Flags::FIN) || l4_packet.flags().contains(Flags::RST);
                                            }
                                            _ => continue,
                                        }
                                        packet_type = PacketType::Tcp;
                                    } else if l3_packet.protocol() == packet::ip::Protocol::Udp {
                                        if self.tcp_only { continue; }
                                        match UdpPacket::no_payload(l3_packet.payload()) {
                                            Ok(p) => l4_packet_udp = p,
                                            _ => continue,
                                        }
                                        packet_type = PacketType::Udp;
                                    } else if l3_packet.protocol() == packet::ip::Protocol::Icmp {
                                        continue;
                                        if self.tcp_only { continue; }
                                        match IcmpPacket::new(l3_packet.payload()) {
                                            Ok(p) => l4_packet_icmp = p,
                                            _ => continue,
                                        }
                                        packet_type = PacketType::Icmp;
                                    } else {
                                        continue;
                                    }
                                },

                                PacketData::L4(_, _) => {
                                    println!("L4 type");
                                    continue;
                                },
                                PacketData::Unsupported(a) => {
                                    println!("Unsupported");
                                    continue;
                                },
                            }

                            let mut packet_to_send = match packet_type {
                                PacketType::Tcp => super::packet::Packet::build_packet_timestamp_size(Parser::octets_to_ip(l3_packet.source().octets()), Parser::octets_to_ip(l3_packet.destination().octets()), l4_packet.source(), l4_packet.destination(), packet::ip::Protocol::Tcp.into(), ts, l3_packet.length() as usize),
                                PacketType::Udp => super::packet::Packet::build_packet_timestamp_size(Parser::octets_to_ip(l3_packet.source().octets()), Parser::octets_to_ip(l3_packet.destination().octets()), l4_packet_udp.source(), l4_packet_udp.destination() , packet::ip::Protocol::Udp.into(), ts, l3_packet.length() as usize),
                                PacketType::Icmp => super::packet::Packet::build_packet_timestamp_size(Parser::octets_to_ip(l3_packet.source().octets()), Parser::octets_to_ip(l3_packet.destination().octets()), 0, 0, packet::ip::Protocol::Icmp.into(), ts, l3_packet.length() as usize),
                            };

                            packet_to_send.has_fin_rst = is_last;
                            packet_to_send.packet_source = PacketSource::FromParser;
                            self.update_statistics(&packet_to_send);
                            self.packet_list.push(packet_to_send);

                            num_packets += 1;
                        }
                        PcapBlock::LegacyHeader(packet_header) => {
                            eprintln!("Read pcap header!");
                            //println!("{:?}", packet_header);
                            trace_linktype = packet_header.network;
                        }
                    }
                }
            },
            _ => { eprintln!("error capture"); }
        }
        println!("Processed packets {}", num_packets);
    }
}
