use std::collections::HashSet;
use std::env;
use std::fs::File;
use pcap_parser::*;
use std::io::Read;
use pcap_parser::data::{get_packetdata, PacketData};
use packet::ether::Packet as EthernetPacket; 
use packet::ip::Packet as IpPacket;
use packet::tcp::Packet as TcpPacket;
pub use packet::udp::Packet as UdpPacket;
use packet::Packet;
pub use pcap_parser::traits::PcapReaderIterator;
use std::io::Write;
use std::net::Ipv4Addr;
use CODA_simu::pcap::IcmpPacket;

enum PacketType {
    Tcp,
    Udp,
    Icmp,
}

fn main() {
    let pcap_filename = &env::args().nth(1).expect("pcap file");
    let output_file = &env::args().nth(2).expect("output file");
    let tcp_only = env::args().nth(3).map_or(false, |arg| arg == "--tcp_only");
    let mut if_linktypes = Vec::new();
    let mut trace_linktype = Linktype(228);
    let mut file = File::open(pcap_filename).unwrap();
    let mut buffer = Vec::new();
    let mut flow_hash_set = HashSet::new();
    let output_dir = std::path::Path::new(output_file).parent().unwrap();
    std::fs::create_dir_all(output_dir).expect("TODO: panic message");
    let mut out_file = File::create(output_file).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    let mut num_packets = 0;
    if tcp_only {
        println!("--tcp_only");
    }
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
                        let l3_packet;
                        let mut l4_packet: TcpPacket<&[u8]> = TcpPacket::unchecked(&[0]);
                        let mut l4_packet_udp: UdpPacket<&[u8]>= UdpPacket::unchecked(&[0]);
                        let mut l4_packet_icmp: IcmpPacket<&[u8]> = IcmpPacket::unchecked(&[0]);
                        let l2_packet;
                        let packet_type;
                        //println!("read packet");
                        match pkt_data {
                            PacketData::L2(a) => {
                                //println!("Ethernet packet");
                                l2_packet = EthernetPacket::new(a).unwrap();
                                if l2_packet.protocol() != packet::ether::Protocol::Ipv4 {
                                    continue;
                                }
                                //unchecked as there's no payload
                                let temp_l3 = IpPacket::unchecked(l2_packet.payload());
                                match temp_l3 {
                                    IpPacket::V4(p) => {
                                        l3_packet = p;
                                    },
                                    _ => {   continue; }
                                }
                                if l3_packet.protocol() == packet::ip::Protocol::Tcp {
                                    //println!("tcp inside ip");
                                    l4_packet = TcpPacket::unchecked(l3_packet.payload());
                                    packet_type = PacketType::Tcp;
                                    //println!("{:?}", l4_packet);
                                } else if l3_packet.protocol() == packet::ip::Protocol::Udp {
                                    if tcp_only { continue; }
                                    if l3_packet.payload().len() < 8 { continue; }
                                    l4_packet_udp = UdpPacket::no_payload(l3_packet.payload()).unwrap();
                                    packet_type = PacketType::Udp;
                                } else if l3_packet.protocol() == packet::ip::Protocol::Icmp {
                                    continue;
                                    if tcp_only { continue; }
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
                                        Ok(p) => l4_packet = p,
                                        _ => continue,
                                    }
                                    packet_type = PacketType::Tcp;
                                } else if l3_packet.protocol() == packet::ip::Protocol::Udp {
                                    if tcp_only { continue; }
                                    match UdpPacket::no_payload(l3_packet.payload()) {
                                        Ok(p) => l4_packet_udp = p,
                                        _ => continue,
                                    }
                                    packet_type = PacketType::Udp;
                                } else if l3_packet.protocol() == packet::ip::Protocol::Icmp {
                                    continue;
                                    if tcp_only { continue; }
                                    match TcpPacket::new(l3_packet.payload()) {
                                        Ok(p) => l4_packet = p,
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

                        let key: (Ipv4Addr, Ipv4Addr, u16, u16, u8) = match packet_type {
                            PacketType::Tcp => (l3_packet.source(), l3_packet.destination(), l4_packet.source(), l4_packet.destination(), packet::ip::Protocol::Tcp.into()),
                            PacketType::Udp => (l3_packet.source(), l3_packet.destination(), l4_packet_udp.source(), l4_packet_udp.destination(), packet::ip::Protocol::Udp.into()),
                            PacketType::Icmp => (l3_packet.source(), l3_packet.destination(), 0u16, 0u16, packet::ip::Protocol::Icmp.into()),
                        };
                        flow_hash_set.insert(key);

                        num_packets += 1;
                        assert_eq!(out_file.write(&key.0.octets()).unwrap(), 4);
                        assert_eq!(out_file.write(&key.1.octets()).unwrap(), 4);
                        assert_eq!(out_file.write(&key.2.to_be_bytes()).unwrap(), 2);
                        assert_eq!(out_file.write(&key.3.to_be_bytes()).unwrap(), 2);
                        assert_eq!(out_file.write(&key.4.to_be_bytes()).unwrap(), 1);
                    }
                    PcapBlock::LegacyHeader(packet_header) => {
                        println!("Read pcap header!");
                        println!("{:?}", packet_header);
                        trace_linktype = packet_header.network;
                    }
                }
            }
        },
        _ => { println!("error capture"); }
    }
    out_file.flush().unwrap();
    //out_file.close().unwrap();
    println!("#packet {}", num_packets);
    println!("#flows {}", flow_hash_set.len());
}
