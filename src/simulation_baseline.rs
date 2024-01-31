use CODA_simu::parser::Parser;
use CODA_simu::cms::CountMinSketch;
use std::collections::{HashMap,HashSet};
use std::rc::Rc;
use std::cell::RefCell;
use std::cmp::max;
use CODA_simu::control_plane::{ControlPlane};
use CODA_simu::packet::{Packet, PacketReceiver};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use clap::Parser as ClapParser;

#[derive(ClapParser)]
struct Args {
    pcap_file: String,
    output_dir: String,
    output_file_name: String,
    cms_rows: usize,
    cms_cols: usize,
    #[clap(short, long, action)]
    tcp_only: bool,
}


fn main() {
    let args = Args::parse();
    let parser = Rc::new(RefCell::new(Parser::build_parser(&args.pcap_file, args.tcp_only)));
    let cms = Rc::new(RefCell::new(CountMinSketch::build_cms(args.cms_rows, args.cms_cols, 16, 24)));
    let control_plane = Rc::new(RefCell::new(ControlPlane::build_control_plane()));


    /* ARCHITECTURE BUILDING */
    parser.borrow_mut().set_next_block(cms.clone());
    cms.borrow_mut().set_control_plane_block(control_plane.clone());


    // compute simulation time 
    let now = Instant::now();

    /* COMMON CODE */
    let mut count = 0;
    let mut concurrent_flows_ctr = 0;
    let mut max_concurrent_flows = 0;
    let mut max_iat= 0.0;
    parser.borrow_mut().parse_pcap();
    let mut pkt_list = parser.borrow().get_packet_list();
    let mut true_stats_per_epoch = HashMap::new();
    let mut flow_first_ts = HashMap::new();
    let mut flow_fifth_ts = HashMap::new();
    let mut flow_last_ts = HashMap::new();
    for pkt in &mut pkt_list {
        count += 1;
        //eprintln!("pkt #{} timestamp {}", count, first_pkt.timestamp);
        if count % 100000 == 0 {
            println!("Packet {}", count);
            println!("Active flows {}", concurrent_flows_ctr);
            println!("Time from start {}", (Instant::now() - now).as_secs_f64());
        }
        let mut nb = parser.borrow_mut().next_block.clone().unwrap();

        // true stats per epoch
        if let Some(v) = true_stats_per_epoch.get_mut(&pkt.get_binary_key()) {
            *v += 1;
            let iat: f64 = pkt.timestamp - flow_last_ts.get(&pkt.get_binary_key()).unwrap();
            max_iat = f64::max(iat, max_iat);
            if *v <= 5 {
                flow_fifth_ts.insert(pkt.get_binary_key(), pkt.timestamp);
            }
            if *v == 5 || (*v < 5 && pkt.has_fin_rst) {
                concurrent_flows_ctr -= 1;
            }
        } else {
            true_stats_per_epoch.insert(pkt.get_binary_key(), 1);
            flow_first_ts.insert(pkt.get_binary_key(), pkt.timestamp);
            flow_fifth_ts.insert(pkt.get_binary_key(), pkt.timestamp);
            if !pkt.has_fin_rst {
                concurrent_flows_ctr += 1;
            }
        }
        flow_last_ts.insert(pkt.get_binary_key(), pkt.timestamp);

        max_concurrent_flows = max(concurrent_flows_ctr, max_concurrent_flows);

        loop {
            let nb_temp = nb.borrow_mut().receive_packet(pkt);
            if nb_temp.is_none() {
                break;
            } else {
                nb = nb_temp.unwrap();
            }
        }
    }
    // end of simulation dump
    std::fs::create_dir_all(format!("{}/cms", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/gt", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/ht", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/pc", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/simulation_configuration", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/cache_spans", args.output_dir)).unwrap();
    let true_stats_write_file = File::create(format!("{}/gt/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let cache_spans_write_file = File::create(format!("{}/cache_spans/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let cms_write_file = File::create(format!("{}/cms/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let simulation_configuration_file = File::create(format!("{}/simulation_configuration/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let mut true_stats_buf_writer = BufWriter::new(true_stats_write_file);
    let mut cache_spans_buf_writer = BufWriter::new(cache_spans_write_file);
    let mut cms_buf_writer = BufWriter::new(cms_write_file);
    let mut simulation_configuration_buf_write = BufWriter::new(simulation_configuration_file);
    for (&k,&v) in &true_stats_per_epoch {
        let string_key = Packet::from_binary_key_to_flow_id_string(k);
        true_stats_buf_writer.write_fmt(format_args!("{},{}\n", string_key, v)).unwrap();
        cache_spans_buf_writer.write_fmt(format_args!("{},{},{}\n", string_key, flow_first_ts.get(&k).unwrap(), flow_fifth_ts.get(&k).unwrap())).unwrap();
    }
    for k in control_plane.borrow().get_keys() {
        let size = cms.borrow().get_size_estimate(k);
        let string_key = Packet::from_binary_key_to_flow_id_string(k);
        cms_buf_writer.write_fmt(format_args!("{},{}\n", string_key, size)).unwrap();
    }

    simulation_configuration_buf_write.write_fmt(format_args!("PCAP file: {}\n", args.pcap_file)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("CMS rows: {}\n", args.cms_rows)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("CMS cols: {}\n", args.cms_cols)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Control Plane accesses: {}\n", control_plane.borrow().get_transmitted_keys_count())).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Max concurrent flows (k=5): {}\n", max_concurrent_flows)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Max IAT: {}\n", max_iat)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("CMS load factor: {}\n", cms.borrow().get_load_factor())).unwrap();
    //simulation_configuration_buf_write.write_fmt(format_args!("CMS column load factor: {}\n", cms.borrow().get_used_columns())).unwrap();

    true_stats_buf_writer.flush().unwrap();
    cache_spans_buf_writer.flush().unwrap();
    cms_buf_writer.flush().unwrap();

    let elapsed_time = Instant::now() - now;
    println!("Simulation time {}", elapsed_time.as_secs_f64());
    println!("CMS load factor: {}", cms.borrow().get_load_factor());
}
