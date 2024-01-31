pub use pcap_parser::traits::PcapReaderIterator;
pub use pcap_parser::*;
use pcap_parser::data::{get_packetdata, PacketData, get_packetdata_ethernet, get_packetdata_ipv4};
pub use std::fs::File;
pub use std::io::ErrorKind;
pub use std::io::Read;
pub use packet::ether::Packet as EthernetPacket; 
pub use packet::ip::Packet as IpPacket;
pub use packet::tcp::Packet as TcpPacket;
pub use packet::tcp::Flags as TcpFlags;
pub use packet::udp::Packet as UdpPacket;
pub use packet::icmp::Packet as IcmpPacket;
pub use packet::Packet;
pub use csv::Writer;
pub use std::io::Write;
//use itertools::Itertools;
//use std::iter;
//
enum PacketType {
    Tcp,
    Udp, 
    Icmp,
}

enum KeySize {
    SrcIp,
    Ips,
    FourTuple,
    FiveTuple
}

pub struct PcapParser<'a> {
    pcap_file: &'a str,
    output_dir: &'a str,
    packet_slot: usize,
}

pub struct PcapKeyExtractor<'a> {
    pcap_file: &'a str,
    output_dir: &'a str,
    key_size: KeySize
}


impl<'a> PcapKeyExtractor<'a> {
    pub fn build_pcap_key_extractor(pcap_file: &'a str, output_dir: &'a str, key_size_num: usize) -> PcapKeyExtractor<'a> {
        let key_size = 
            if key_size_num == 1 {
                KeySize::SrcIp
            } else if key_size_num == 2 {
                KeySize::Ips
            } else if key_size_num == 4 {
                KeySize::FourTuple
            } else {
                KeySize::FiveTuple
            };
        let pp = PcapKeyExtractor{
            pcap_file,
            output_dir,
            key_size 
        };
        pp
    }


    pub fn read(&self) {
        let mut if_linktypes = Vec::new();
        let mut trace_linktype = Linktype(228);
        let mut file = File::open(self.pcap_file).unwrap();
        let mut buffer = Vec::new();
        let mut key_vec = vec![];
        let mut ts_vec = vec![];
        std::fs::create_dir_all(self.output_dir);
        let mut wtr_keys = File::create(format!("{}/{}",self.output_dir,"/keys.csv")).unwrap();
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
                            let l4_packet;
                            let l2_packet; 
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
                                        l4_packet = TcpPacket::new(l3_packet.payload()).unwrap();
                                        //println!("{:?}", l4_packet);
                                    } else {
                                        //println!("not tcp {:?}", l3_packet.protocol());
                                        println!("not tcp");
                                        l4_packet = TcpPacket::new(l3_packet.payload()).unwrap();
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
                                        //println!("tcp inside ip");
                                        match TcpPacket::new(l3_packet.payload()) {
                                            Ok(p) => l4_packet = p,
                                            _ => continue,
                                        }
                                        //println!("{:?}", l4_packet);
                                    } else {
                                        //println!("not tcp {:?}", l3_packet.protocol());
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
                            let mut key  = u32::from_ne_bytes(l3_packet.source().octets()) as u128;
                            key = (key << 32) |  u32::from_ne_bytes(l3_packet.destination().octets()) as u128;
                            key = (key << 16) |  (l4_packet.source() as u128);
                            key = (key << 16) |  (l4_packet.destination() as u128);
                            key = (key << 8) | 6;

                            match self.key_size {
                                KeySize::SrcIp => {key = key >> 72;}
                                KeySize::Ips =>  {key = key >> 40;}
                                KeySize::FourTuple => {key = key >> 8;}
                                _ => ()
                            }

                            ts_vec.push(ts);
                            key_vec.push(key);
                            num_packets += 1;

                            //println!("key {:?}", key);
                            //println!("counter {}", counter);

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
        println!("#packet {}", num_packets);
        for (i, &k) in key_vec.iter().enumerate() {
            wtr_keys.write_fmt(format_args!("{} {}\n",ts_vec[i], k)).unwrap();
        }
        wtr_keys.flush().unwrap();
    }
}

impl<'a> PcapParser<'a> {
    pub fn build_pcap_parser(pcap_file: &'a str, output_dir: &'a str, packet_slot: usize) -> PcapParser<'a> {
        let pp = PcapParser {
            pcap_file,
            output_dir,
            packet_slot
        };
        pp
    }

    pub fn read(&self) {
        let mut if_linktypes = Vec::new();
        let mut trace_linktype = Linktype(228);
        let mut file = File::open(self.pcap_file).unwrap();
        let mut buffer = Vec::new();
        //TCP
        let mut timestamps_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<f64>>::new();
        let mut interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<f64>>::new();
        let mut packets_size_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<u16>>::new();
        let mut max_interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut min_interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut max_timestamp_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut min_timestamp_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut flags_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<TcpFlags>>::new(); 
        let mut oracle_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), (f64, f64, f64, f64, usize)>::new();
        //UDP
        let mut udp_timestamps_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<f64>>::new();
        let mut udp_interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<f64>>::new();
        let mut udp_packets_size_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<u16>>::new();
        let mut udp_max_interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut udp_min_interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut udp_max_timestamp_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut udp_min_timestamp_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut udp_flags_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<TcpFlags>>::new(); 
        let mut udp_oracle_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), (f64, f64, f64, f64, usize)>::new();
        //ICMP
        let mut icmp_timestamps_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<f64>>::new();
        let mut icmp_interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<f64>>::new();
        let mut icmp_packets_size_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<u16>>::new();
        let mut icmp_max_interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut icmp_min_interarrival_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut icmp_max_timestamp_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut icmp_min_timestamp_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), f64>::new(); 
        let mut icmp_flags_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<TcpFlags>>::new(); 
        let mut icmp_oracle_hashmap = std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), (f64, f64, f64, f64, usize)>::new();
        std::fs::create_dir_all(self.output_dir);
        //TCP
        let mut wtr_timestamps = File::create(format!("{}/{}",self.output_dir,"/timestamps_tcp.csv")).unwrap();
        let mut wtr_interarrival  = File::create(format!("{}/{}",self.output_dir,"/interarrival_tcp.csv")).unwrap();
        let mut wtr_oracle  = File::create(format!("{}/{}",self.output_dir,"/oracle_tcp.csv")).unwrap();
        //UDP
        let mut udp_wtr_timestamps = File::create(format!("{}/{}",self.output_dir,"/timestamps_udp.csv")).unwrap();
        let mut udp_wtr_interarrival  = File::create(format!("{}/{}",self.output_dir,"/interarrival_udp.csv")).unwrap();
        let mut udp_wtr_oracle  = File::create(format!("{}/{}",self.output_dir,"/oracle_udp.csv")).unwrap();
        //ICMP
        let mut icmp_wtr_timestamps = File::create(format!("{}/{}",self.output_dir,"/timestamps_icmp.csv")).unwrap();
        let mut icmp_wtr_interarrival  = File::create(format!("{}/{}",self.output_dir,"/interarrival_icmp.csv")).unwrap();
        let mut icmp_wtr_oracle  = File::create(format!("{}/{}",self.output_dir,"/oracle_icmp.csv")).unwrap();
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
                            //println!("read packet");
                            match pkt_data {
                                PacketData::L2(a) => {
                                    //println!("Ethernet packet");
                                    l2_packet = EthernetPacket::new(a).unwrap();
                                    //unchecked as there's no payload
                                    let mut payload = l2_packet.payload();
                                    if l2_packet.protocol() == packet::ether::Protocol::Vlan {
                                        payload = &payload[4..];
                                    } else if l2_packet.protocol() != packet::ether::Protocol::Ipv4 {
                                        continue;
                                    }
                                    let temp_l3 = IpPacket::unchecked(payload);
                                    match temp_l3 {
                                        IpPacket::V4(p) => {
                                            l3_packet = p;
                                        },
                                        _ => {   continue; }
                                    }
                                    if l3_packet.payload().len() == 0 {
                                        continue;
                                    }
                                    if l3_packet.protocol() == packet::ip::Protocol::Tcp {
                                        //println!("tcp inside ip");
                                        l4_packet = TcpPacket::unchecked(l3_packet.payload());
                                        packet_type = PacketType::Tcp;
                                        //println!("{:?}", l4_packet);
                                    } else if l3_packet.protocol() == packet::ip::Protocol::Udp {
                                        //println!("not tcp {:?}", l3_packet.protocol());
                                        //println!("not tcp");
                                        //println!("udp packet length {}", l3_packet.payload().len());
                                        l4_packet_udp = UdpPacket::no_payload(l3_packet.payload()).unwrap();
                                        packet_type = PacketType::Udp;
                                    } else if l3_packet.protocol() == packet::ip::Protocol::Icmp {
                                        l4_packet_icmp = IcmpPacket::unchecked(l3_packet.payload());
                                        packet_type = PacketType::Icmp;
                                    } else {
                                        continue;
                                    }

                                    //if l3_packet.source() == Ipv4Addr::new(120,177,135,22) && 
                                    //    l3_packet.destination() == Ipv4Addr::new(202,22,211,5) &&
                                    //    l4_packet.source() == 34658 &&
                                    //    l4_packet.destination() == 80 {
                                    //    println!("found packet");
                                    //}

                                },
                                PacketData::L3(_, b) => {
                                    //match IpPacket::new(b) {
                                    //    Ok(p) => match p { 
                                    //        IpPacket::V4(p) => l3_packet = p,
                                    //        _ => { continue;},
                                    //    },
                                    //    _ => { continue;},
                                    //}
                                    //println!("l3 packet");
                                    let temp_l3 = IpPacket::unchecked(b);
                                    match temp_l3 {
                                        IpPacket::V4(p) => {l3_packet = p; },
                                        _ => { continue; }

                                    }
                                    if l3_packet.payload().len() == 0 {
                                        continue;
                                    }
                                    if l3_packet.protocol() == packet::ip::Protocol::Tcp {
                                        //println!("tcp inside ip");
                                        //println!("tcp packet length {}", l3_packet.payload().len());
                                        //l4_packet = TcpPacket::new(l3_packet.payload()).unwrap();
                                        l4_packet = TcpPacket::unchecked(l3_packet.payload());
                                        packet_type = PacketType::Tcp;
                                        //println!("{:?}", l4_packet);
                                    } else if l3_packet.protocol() == packet::ip::Protocol::Udp {
                                        //println!("not tcp {:?}", l3_packet.protocol());
                                        //println!("not tcp");
                                        //println!("udp packet length {}", l3_packet.payload().len());
                                        l4_packet_udp = UdpPacket::no_payload(l3_packet.payload()).unwrap();
                                        packet_type = PacketType::Udp;
                                    } else if l3_packet.protocol() == packet::ip::Protocol::Icmp {
                                        l4_packet_icmp = IcmpPacket::unchecked(l3_packet.payload());
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
                            let mut key;
                            match packet_type {
                                PacketType::Tcp => {key  = (l3_packet.source(), l3_packet.destination(), l4_packet.source(), l4_packet.destination());},
                                PacketType::Udp => {key  = (l3_packet.source(), l3_packet.destination(), l4_packet_udp.source(), l4_packet_udp.destination());},
                                PacketType::Icmp => {key  = (l3_packet.source(), l3_packet.destination(), 0, 0);},
                            }
                            num_packets += 1;

                            match packet_type {
                                PacketType::Tcp => {
                                    if !timestamps_hashmap.contains_key(&key) {
                                        timestamps_hashmap.insert(key, vec![ts]);
                                        min_timestamp_hashmap.insert(key, ts);
                                        max_timestamp_hashmap.insert(key, ts);
                                        flags_hashmap.insert(key, vec![l4_packet.flags()]);
                                        packets_size_hashmap.insert(key, vec![l3_packet.length()]);
                                    } else {
                                        // update timestamps hashmap 
                                        let old_vec = timestamps_hashmap.entry(key).or_insert(vec![0.0]);
                                        let gap = ts - old_vec.last().copied().unwrap();
                                        old_vec.push(ts);
                                        //update flags hashmap
                                        let old_flags = flags_hashmap.entry(key).or_default();
                                        old_flags.push(l4_packet.flags());
                                        //update packet length hashmap 
                                        let old_flags = packets_size_hashmap.entry(key).or_default();
                                        old_flags.push(l3_packet.length());
                                        // update max min timestamps hashmap
                                        let min_entry = min_timestamp_hashmap.entry(key).or_insert(0.0);
                                        let max_entry = max_timestamp_hashmap.entry(key).or_insert(100.0);
                                        if ts < *min_entry {
                                            *min_entry = ts;
                                        }
                                        if ts > *max_entry {
                                            *max_entry = ts;
                                        }
                                        
                                        // update interarrival
                                        if !interarrival_hashmap.contains_key(&key) {
                                            interarrival_hashmap.insert(key, vec![gap]);
                                            min_interarrival_hashmap.insert(key, gap);
                                            max_interarrival_hashmap.insert(key, gap);
                                        } else {
                                            let old_vec = interarrival_hashmap.entry(key).or_insert(vec![0.0]);
                                            old_vec.push(gap);
                                            // update max min interarrival hashmap
                                            let min_entry = min_interarrival_hashmap.entry(key).or_insert(0.0);
                                            let max_entry = max_interarrival_hashmap.entry(key).or_insert(100.0);
                                            if gap < *min_entry {
                                                *min_entry = gap;
                                            }
                                            if gap > *max_entry {
                                                *max_entry = gap;
                                            }
                                        }
                                    }
                                }
                                PacketType::Udp => {
                                    if !udp_timestamps_hashmap.contains_key(&key) {
                                        udp_timestamps_hashmap.insert(key, vec![ts]);
                                        udp_min_timestamp_hashmap.insert(key, ts);
                                        udp_max_timestamp_hashmap.insert(key, ts);
                                        udp_flags_hashmap.insert(key, vec![]);
                                        udp_packets_size_hashmap.insert(key, vec![l3_packet.length()]);
                                    } else {
                                        // update timestamps hashmap 
                                        let old_vec = udp_timestamps_hashmap.entry(key).or_insert(vec![0.0]);
                                        let gap = ts - old_vec.last().copied().unwrap();
                                        old_vec.push(ts);
                                        //update flags hashmap
                                        //let old_flags = udp_flags_hashmap.entry(key).or_default();
                                        //old_flags.push(l4_packet.flags());
                                        //update packet length hashmap 
                                        let old_flags = udp_packets_size_hashmap.entry(key).or_default();
                                        old_flags.push(l3_packet.length());
                                        // update max min timestamps hashmap
                                        let min_entry = udp_min_timestamp_hashmap.entry(key).or_insert(0.0);
                                        let max_entry = udp_max_timestamp_hashmap.entry(key).or_insert(100.0);
                                        if ts < *min_entry {
                                            *min_entry = ts;
                                        }
                                        if ts > *max_entry {
                                            *max_entry = ts;
                                        }
                                        
                                        // update interarrival
                                        if !udp_interarrival_hashmap.contains_key(&key) {
                                            udp_interarrival_hashmap.insert(key, vec![gap]);
                                            udp_min_interarrival_hashmap.insert(key, gap);
                                            udp_max_interarrival_hashmap.insert(key, gap);
                                        } else {
                                            let old_vec = udp_interarrival_hashmap.entry(key).or_insert(vec![0.0]);
                                            old_vec.push(gap);
                                            // update max min interarrival hashmap
                                            let min_entry = udp_min_interarrival_hashmap.entry(key).or_insert(0.0);
                                            let max_entry = udp_max_interarrival_hashmap.entry(key).or_insert(100.0);
                                            if gap < *min_entry {
                                                *min_entry = gap;
                                            }
                                            if gap > *max_entry {
                                                *max_entry = gap;
                                            }
                                        }
                                    }
                                }

                                PacketType::Icmp => {
                                    if !icmp_timestamps_hashmap.contains_key(&key) {
                                        icmp_timestamps_hashmap.insert(key, vec![ts]);
                                        icmp_min_timestamp_hashmap.insert(key, ts);
                                        icmp_max_timestamp_hashmap.insert(key, ts);
                                        icmp_flags_hashmap.insert(key, vec![]);
                                        icmp_packets_size_hashmap.insert(key, vec![l3_packet.length()]);
                                    } else {
                                        // update timestamps hashmap 
                                        let old_vec = icmp_timestamps_hashmap.entry(key).or_insert(vec![0.0]);
                                        let gap = ts - old_vec.last().copied().unwrap();
                                        old_vec.push(ts);
                                        //update flags hashmap
                                        //let old_flags = icmp_flags_hashmap.entry(key).or_default();
                                        //old_flags.push(l4_packet.flags());
                                        //update packet length hashmap 
                                        let old_flags = icmp_packets_size_hashmap.entry(key).or_default();
                                        old_flags.push(l3_packet.length());
                                        // update max min timestamps hashmap
                                        let min_entry = icmp_min_timestamp_hashmap.entry(key).or_insert(0.0);
                                        let max_entry = icmp_max_timestamp_hashmap.entry(key).or_insert(100.0);
                                        if ts < *min_entry {
                                            *min_entry = ts;
                                        }
                                        if ts > *max_entry {
                                            *max_entry = ts;
                                        }
                                        
                                        // update interarrival
                                        if !icmp_interarrival_hashmap.contains_key(&key) {
                                            icmp_interarrival_hashmap.insert(key, vec![gap]);
                                            icmp_min_interarrival_hashmap.insert(key, gap);
                                            icmp_max_interarrival_hashmap.insert(key, gap);
                                        } else {
                                            let old_vec = icmp_interarrival_hashmap.entry(key).or_insert(vec![0.0]);
                                            old_vec.push(gap);
                                            // update max min interarrival hashmap
                                            let min_entry = icmp_min_interarrival_hashmap.entry(key).or_insert(0.0);
                                            let max_entry = icmp_max_interarrival_hashmap.entry(key).or_insert(100.0);
                                            if gap < *min_entry {
                                                *min_entry = gap;
                                            }
                                            if gap > *max_entry {
                                                *max_entry = gap;
                                            }
                                        }
                                    }
                                }

                            }
                            //println!("{}", ts);

                            //println!("key {:?}", key);
                            //println!("counter {}", counter);

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
        println!("#packet {}", num_packets);


        //timestamps_hashmap.iter_mut().for_each(|(_, v)| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); } );
        // create oracle 
        // TCP 
        for (k,v) in min_timestamp_hashmap {
            let min_ts = v;
            let max_ts = max_timestamp_hashmap.get(&k).unwrap();
            let min_gap = min_interarrival_hashmap.get(&k).or_else(|| {Some(&-1.0)}).unwrap();
            let max_gap = max_interarrival_hashmap.get(&k).or_else(|| {Some(&-1.0)}).unwrap();
            let vector = timestamps_hashmap.get(&k).unwrap();
            oracle_hashmap.insert(k, (*min_gap, *max_gap, min_ts, *max_ts, vector.len()));
        }
        let new_timestamp_hashmap : std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<String>>  = timestamps_hashmap.into_iter().map(|(k, v)| { (k, v.iter().map(|i| i.to_string()).collect::<Vec<String>>()) } ).collect();
        let new_interarrival_hashmap : std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<String>>  = interarrival_hashmap.into_iter().map(|(k, v)| { (k, v.iter().map(|i| i.to_string()).collect::<Vec<String>>()) } ).collect();
        for (k,v) in new_timestamp_hashmap {
            wtr_timestamps.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,6)).unwrap();
            let temp_v = v.iter().map(|x| ((x.parse::<f64>().unwrap()*1000000.0).round())/1000000.0).map(|x| x.to_string() ).collect::<Vec<String>>();
            wtr_timestamps.write(temp_v.join(",").as_bytes()).unwrap();
            wtr_timestamps.write(b"\n").unwrap();
        }
        wtr_timestamps.flush().unwrap();
        for (k,v) in &new_interarrival_hashmap {
            wtr_interarrival.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,6)).unwrap();
            let temp_v = v.iter().map(|x| ((x.parse::<f64>().unwrap()*1000000.0).round())/1000000.0).map(|x| x.to_string() ).collect::<Vec<String>>();
            wtr_interarrival.write(temp_v.join(",").as_bytes()).unwrap();
            wtr_interarrival.write(b"\n").unwrap();
        }
        wtr_interarrival.flush().unwrap();
        let mut sorted_oracle : Vec<_> = oracle_hashmap.iter().collect();
        sorted_oracle.sort_by(|x,y| x.1.2.partial_cmp(&(y.1.2)).unwrap());
        for (k,v) in sorted_oracle  {
            wtr_oracle.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,6)).unwrap();
            wtr_oracle.write_fmt(format_args!("{:.6},{:.6},{:.6},{:.6},{},", v.0, v.1, v.2, v.3, v.4)).unwrap();


            //write iat list
            let dummy_vec = vec![];
            let temp_iat_vec = new_interarrival_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_iat_vec_lenght = temp_iat_vec.len();
            for i in 0..(self.packet_slot-1) {
                if i >= temp_iat_vec_lenght {
                    wtr_oracle.write_fmt(format_args!(",")).unwrap();
                } else {
                    wtr_oracle.write_fmt(format_args!("{},", temp_iat_vec[i] )).unwrap();
                }
            }
            //write flag list
            let dummy_vec = vec![];
            let temp_flags_vec = flags_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_flags_vec_lenght = temp_flags_vec.len();
            for i in 0..self.packet_slot {
                if i >= temp_flags_vec_lenght {
                    wtr_oracle.write_fmt(format_args!(",")).unwrap();
                } else {
                    wtr_oracle.write_fmt(format_args!("{},", temp_flags_vec[i].bits() )).unwrap();
                }
            }
            //write packet size list
            let dummy_vec = vec![];
            let temp_size_vec = packets_size_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_size_vec_lenght = temp_size_vec.len();
            for i in 0..self.packet_slot {
                if i >= temp_size_vec_lenght {
                    if i == self.packet_slot - 1 {
                        //wtr_oracle.write_fmt(format_args!("{}", temp_size_vec[i] )).unwrap();
                    } else {
                        wtr_oracle.write_fmt(format_args!(",")).unwrap();
                    }
                } else {
                    if i == self.packet_slot - 1 {
                        wtr_oracle.write_fmt(format_args!("{}", temp_size_vec[i] )).unwrap();
                    } else {
                        wtr_oracle.write_fmt(format_args!("{},", temp_size_vec[i] )).unwrap();
                    }
                }
            }
            wtr_oracle.write(b"\n").unwrap();
        }
        wtr_oracle.flush().unwrap();
        //println!("{:?}", new_hashmap);
        
        // UDP 
        for (k,v) in udp_min_timestamp_hashmap {
            let min_ts = v;
            let max_ts = udp_max_timestamp_hashmap.get(&k).unwrap();
            let min_gap = udp_min_interarrival_hashmap.get(&k).or_else(|| {Some(&-1.0)}).unwrap();
            let max_gap = udp_max_interarrival_hashmap.get(&k).or_else(|| {Some(&-1.0)}).unwrap();
            let vector = udp_timestamps_hashmap.get(&k).unwrap();
            udp_oracle_hashmap.insert(k, (*min_gap, *max_gap, min_ts, *max_ts, vector.len()));
        }
        let udp_new_timestamp_hashmap : std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<String>>  = udp_timestamps_hashmap.into_iter().map(|(k, v)| { (k, v.iter().map(|i| i.to_string()).collect::<Vec<String>>()) } ).collect();
        let udp_new_interarrival_hashmap : std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<String>>  = udp_interarrival_hashmap.into_iter().map(|(k, v)| { (k, v.iter().map(|i| i.to_string()).collect::<Vec<String>>()) } ).collect();
        for (k,v) in udp_new_timestamp_hashmap {
            udp_wtr_timestamps.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,17)).unwrap();
            let temp_v = v.iter().map(|x| ((x.parse::<f64>().unwrap()*1000000.0).round())/1000000.0).map(|x| x.to_string() ).collect::<Vec<String>>();
            udp_wtr_timestamps.write(temp_v.join(",").as_bytes()).unwrap();
            udp_wtr_timestamps.write(b"\n").unwrap();
        }
        udp_wtr_timestamps.flush().unwrap();
        for (k,v) in &udp_new_interarrival_hashmap {
            udp_wtr_interarrival.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,17)).unwrap();
            let temp_v = v.iter().map(|x| ((x.parse::<f64>().unwrap()*1000000.0).round())/1000000.0).map(|x| x.to_string() ).collect::<Vec<String>>();
            udp_wtr_interarrival.write(temp_v.join(",").as_bytes()).unwrap();
            udp_wtr_interarrival.write(b"\n").unwrap();
        }
        udp_wtr_interarrival.flush().unwrap();
        let mut udp_sorted_oracle : Vec<_> = udp_oracle_hashmap.iter().collect();
        udp_sorted_oracle.sort_by(|x,y| x.1.2.partial_cmp(&(y.1.2)).unwrap());
        for (k,v) in udp_sorted_oracle  {
            udp_wtr_oracle.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,17)).unwrap();
            udp_wtr_oracle.write_fmt(format_args!("{:.6},{:.6},{:.6},{:.6},{},", v.0, v.1, v.2, v.3, v.4)).unwrap();


            //write iat list
            let dummy_vec = vec![];
            let temp_iat_vec = udp_new_interarrival_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_iat_vec_lenght = temp_iat_vec.len();
            for i in 0..(self.packet_slot-1) {
                if i >= temp_iat_vec_lenght {
                    udp_wtr_oracle.write_fmt(format_args!(",")).unwrap();
                } else {
                    udp_wtr_oracle.write_fmt(format_args!("{},", temp_iat_vec[i] )).unwrap();
                }
            }
            //write flag list
            let dummy_vec = vec![];
            let temp_flags_vec = udp_flags_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_flags_vec_lenght = temp_flags_vec.len();
            for i in 0..self.packet_slot {
                if i >= temp_flags_vec_lenght {
                    udp_wtr_oracle.write_fmt(format_args!(",")).unwrap();
                } else {
                    udp_wtr_oracle.write_fmt(format_args!("{},", temp_flags_vec[i].bits() )).unwrap();
                }
            }
            //write packet size list
            let dummy_vec = vec![];
            let temp_size_vec = udp_packets_size_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_size_vec_lenght = temp_size_vec.len();
            for i in 0..self.packet_slot {
                if i >= temp_size_vec_lenght {
                    if i == self.packet_slot - 1 {
                        //wtr_oracle.write_fmt(format_args!("{}", temp_size_vec[i] )).unwrap();
                    } else {
                        udp_wtr_oracle.write_fmt(format_args!(",")).unwrap();
                    }
                } else {
                    if i == self.packet_slot - 1 {
                        udp_wtr_oracle.write_fmt(format_args!("{}", temp_size_vec[i] )).unwrap();
                    } else {
                        udp_wtr_oracle.write_fmt(format_args!("{},", temp_size_vec[i] )).unwrap();
                    }
                }
            }
            udp_wtr_oracle.write(b"\n").unwrap();
        }
        udp_wtr_oracle.flush().unwrap();

        // ICMP
        for (k,v) in icmp_min_timestamp_hashmap {
            let min_ts = v;
            let max_ts = icmp_max_timestamp_hashmap.get(&k).unwrap();
            let min_gap = icmp_min_interarrival_hashmap.get(&k).or_else(|| {Some(&-1.0)}).unwrap();
            let max_gap = icmp_max_interarrival_hashmap.get(&k).or_else(|| {Some(&-1.0)}).unwrap();
            let vector = icmp_timestamps_hashmap.get(&k).unwrap();
            icmp_oracle_hashmap.insert(k, (*min_gap, *max_gap, min_ts, *max_ts, vector.len()));
        }
        let icmp_new_timestamp_hashmap : std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<String>>  = icmp_timestamps_hashmap.into_iter().map(|(k, v)| { (k, v.iter().map(|i| i.to_string()).collect::<Vec<String>>()) } ).collect();
        let icmp_new_interarrival_hashmap : std::collections::HashMap::<(std::net::Ipv4Addr, std::net::Ipv4Addr, u16, u16), Vec<String>>  = icmp_interarrival_hashmap.into_iter().map(|(k, v)| { (k, v.iter().map(|i| i.to_string()).collect::<Vec<String>>()) } ).collect();
        for (k,v) in icmp_new_timestamp_hashmap {
            icmp_wtr_timestamps.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,1)).unwrap();
            let temp_v = v.iter().map(|x| ((x.parse::<f64>().unwrap()*1000000.0).round())/1000000.0).map(|x| x.to_string() ).collect::<Vec<String>>();
            icmp_wtr_timestamps.write(temp_v.join(",").as_bytes()).unwrap();
            icmp_wtr_timestamps.write(b"\n").unwrap();
        }
        icmp_wtr_timestamps.flush().unwrap();
        for (k,v) in &icmp_new_interarrival_hashmap {
            icmp_wtr_interarrival.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,1)).unwrap();
            let temp_v = v.iter().map(|x| ((x.parse::<f64>().unwrap()*1000000.0).round())/1000000.0).map(|x| x.to_string() ).collect::<Vec<String>>();
            icmp_wtr_interarrival.write(temp_v.join(",").as_bytes()).unwrap();
            icmp_wtr_interarrival.write(b"\n").unwrap();
        }
        icmp_wtr_interarrival.flush().unwrap();
        let mut icmp_sorted_oracle : Vec<_> = icmp_oracle_hashmap.iter().collect();
        icmp_sorted_oracle.sort_by(|x,y| x.1.2.partial_cmp(&(y.1.2)).unwrap());
        for (k,v) in icmp_sorted_oracle  {
            icmp_wtr_oracle.write_fmt(format_args!("{},{},{},{},{},",k.0,k.1,k.2,k.3,1)).unwrap();
            icmp_wtr_oracle.write_fmt(format_args!("{:.6},{:.6},{:.6},{:.6},{},", v.0, v.1, v.2, v.3, v.4)).unwrap();


            //write iat list
            let dummy_vec = vec![];
            let temp_iat_vec = icmp_new_interarrival_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_iat_vec_lenght = temp_iat_vec.len();
            for i in 0..(self.packet_slot-1) {
                if i >= temp_iat_vec_lenght {
                    icmp_wtr_oracle.write_fmt(format_args!(",")).unwrap();
                } else {
                    icmp_wtr_oracle.write_fmt(format_args!("{},", temp_iat_vec[i] )).unwrap();
                }
            }
            //write flag list
            let dummy_vec = vec![];
            let temp_flags_vec = icmp_flags_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_flags_vec_lenght = temp_flags_vec.len();
            for i in 0..self.packet_slot {
                if i >= temp_flags_vec_lenght {
                    icmp_wtr_oracle.write_fmt(format_args!(",")).unwrap();
                } else {
                    icmp_wtr_oracle.write_fmt(format_args!("{},", temp_flags_vec[i].bits() )).unwrap();
                }
            }
            //write packet size list
            let dummy_vec = vec![];
            let temp_size_vec = icmp_packets_size_hashmap.get(k).unwrap_or(&dummy_vec);
            let temp_size_vec_lenght = temp_size_vec.len();
            for i in 0..self.packet_slot {
                if i >= temp_size_vec_lenght {
                    if i == self.packet_slot - 1 {
                        //wtr_oracle.write_fmt(format_args!("{}", temp_size_vec[i] )).unwrap();
                    } else {
                        icmp_wtr_oracle.write_fmt(format_args!(",")).unwrap();
                    }
                } else {
                    if i == self.packet_slot - 1 {
                        icmp_wtr_oracle.write_fmt(format_args!("{}", temp_size_vec[i] )).unwrap();
                    } else {
                        icmp_wtr_oracle.write_fmt(format_args!("{},", temp_size_vec[i] )).unwrap();
                    }
                }
            }
            icmp_wtr_oracle.write(b"\n").unwrap();
        }
        icmp_wtr_oracle.flush().unwrap();
    }
}
