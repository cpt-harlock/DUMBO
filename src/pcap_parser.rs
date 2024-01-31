use CODA_simu::pcap::PcapParser;
use std::env;


fn main() {
    let pcap_filename = &env::args().nth(1).expect("pcap file");
    let output_dir = &env::args().nth(2).expect("output_dir");
    let packet_count = env::args().nth(3).expect("packet_count").parse::<usize>().unwrap();
    let parser = PcapParser::build_pcap_parser(pcap_filename, output_dir, packet_count);
    parser.read();
}

