#![feature(trait_upcasting)]    // TODO remove this once feature is merged in stable (https://github.com/rust-lang/rust/pull/118133)

use CODA_simu::parser::Parser;
use CODA_simu::hash_table::HashTable;
use CODA_simu::model_wrapper::{ModelWrapper};
use CODA_simu::flow_manager::{FlowManager};
use CODA_simu::cms::CountMinSketch;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use CODA_simu::bloom_filter::MiceBloomFilter;
use CODA_simu::control_plane::{ControlPlane};
use CODA_simu::packet::{Packet, PacketReceiver};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use clap::Parser as ClapParser;
use std::str::FromStr;
use CODA_simu::model_wrapper::constant::ModelWrapperConstant;
use CODA_simu::model_wrapper::five_tuple_aggr::ModelWrapperFiveTupleAggr;
use CODA_simu::model_wrapper::oracle::ModelWrapperOracle;
use CODA_simu::model_wrapper::random::ModelWrapperFiveTupleAggrRandom;
use CODA_simu::model_wrapper::simulated_ap::ModelWrapperSimulatedAP;
use CODA_simu::model_wrapper::simulated_rates::ModelWrapperSimulatedRates;
use CODA_simu::flow_manager::fm_evict_oldest::FlowManagerEvictOldest;
use CODA_simu::flow_manager::fm_evict_timeout::FlowManagerEvictTimeout;
use CODA_simu::flow_manager::fm_infinite::InfiniteFlowManager;
use CODA_simu::flow_manager::fm_hierarchical::HierarchicalFlowManager;
use CODA_simu::model_wrapper::onnx_pre_bins::ModelWrapperOnnxPreBins;


#[derive(clap::ValueEnum, Clone)]
enum ModelType {
    Oracle,
    Random,
    Constant,
    SynthAp,
    SynthRates,
    FiveTupleAggr,
    OnnxPreBins,
}

#[derive(clap::ValueEnum, Clone)]
enum FlowManagerType {
    EvictOldest,
    EvictTimeout,
    Infinite,
    Layered
}

#[derive(ClapParser)]
struct Args {
    #[clap(value_enum)]
    model_type: ModelType,
    pcap_file: String,
    model_file: String,
    features_file: String,
    bins_file: String,
    output_dir: String,
    output_file_name: String,
    ht_rows: usize,
    ht_slots: usize,
    ht_count: usize,
    cms_rows: usize,
    cms_cols: usize,
    #[clap(value_enum)]
    flow_manager_type: FlowManagerType,
    flow_manager_rows: usize,
    flow_manager_slots: usize,
    flow_manager_slot_size: usize,
    flow_manager_packet_limit: usize,
    flow_manager_hierarchy: String,
    bf_size: usize,
    bf_hash_count: usize, 
    model_threshold: f64,
    model_ap: f64,
    model_emr: f64,
    model_mmr: f64,
    #[clap(short, long, action)]
    tcp_only: bool,
}


fn main() {
    let mut now;
    let mut elapsed_time;
    let args = Args::parse();
    //let sink = Rc::new(RefCell::new(PacketSink {}));
    let parser = Rc::new(RefCell::new(Parser::build_parser(&args.pcap_file, args.tcp_only)));
    let hash_table = Rc::new(RefCell::new(HashTable::<u128,usize>::build_hash_table(args.ht_rows, args.ht_slots, 17, args.ht_count)));
    let cms = Rc::new(RefCell::new(CountMinSketch::build_cms(args.cms_rows, args.cms_cols, 16, 24)));
    let bf = Rc::new(RefCell::new(MiceBloomFilter::build_mice_bloom_filter(args.bf_size, args.bf_hash_count)));
    let control_plane = Rc::new(RefCell::new(ControlPlane::build_control_plane()));

    // Flow Manager (former packet cache)
    let flow_manager: Rc<RefCell<dyn FlowManager>> = match args.flow_manager_type {
        FlowManagerType::EvictOldest => { Rc::new(RefCell::new(FlowManagerEvictOldest::build(args.flow_manager_rows, args.flow_manager_slots, args.flow_manager_packet_limit, 323))) }
        FlowManagerType::EvictTimeout => { Rc::new(RefCell::new(FlowManagerEvictTimeout::build(args.flow_manager_rows, args.flow_manager_slots, args.flow_manager_packet_limit, 323, 2000.0))) }
        FlowManagerType::Infinite => { Rc::new(RefCell::new(InfiniteFlowManager::build(args.flow_manager_packet_limit))) }
        FlowManagerType::Layered => {
            //let proportions = vec![1.0, 0.85, 0.7, 0.62];
            //let proportions = vec![1.0, 0.7144182260295988, 0.578189415431055, 0.508657433933787];
            let proportions: Vec<f64> = args.flow_manager_hierarchy.split_whitespace().map(|s| s.parse().unwrap()).collect();
            let mut real_proportions = vec![];
            for i in 0..(args.flow_manager_packet_limit -1) {
                real_proportions.push(proportions[i]);
            }
            let sum: f64 = real_proportions.iter().sum();
            real_proportions = real_proportions.iter_mut().map(|x| { *x/sum }).collect();
            Rc::new(RefCell::new(HierarchicalFlowManager::build(args.flow_manager_packet_limit, real_proportions, 323, args.flow_manager_slots, args.flow_manager_rows)))
        }
    };

    // Model
    now = Instant::now();
    let model_wrapper: Rc<RefCell<dyn ModelWrapper>> = match args.model_type {
        ModelType::Oracle => { Rc::new(RefCell::new(ModelWrapperOracle::build(args.model_file.clone()))) }
        ModelType::Random => { Rc::new(RefCell::new(ModelWrapperFiveTupleAggrRandom::build())) }
        ModelType::Constant => { Rc::new(RefCell::new(ModelWrapperConstant::build())) }
        ModelType::FiveTupleAggr => { Rc::new(RefCell::new(ModelWrapperFiveTupleAggr::build(args.model_file.clone(), args.model_threshold, args.features_file.clone()))) }
        ModelType::OnnxPreBins => { Rc::new(RefCell::new(ModelWrapperOnnxPreBins::build(args.model_file.clone(), args.model_threshold, args.features_file.clone(), args.bins_file))) }
        ModelType::SynthAp => { Rc::new(RefCell::new(ModelWrapperSimulatedAP::build(args.model_file.clone(), args.model_ap, args.model_threshold))) }
        ModelType::SynthRates => { Rc::new(RefCell::new(ModelWrapperSimulatedRates::build(args.model_file.clone(), args.model_emr, args.model_mmr))) }
    };
    elapsed_time = Instant::now() - now;
    println!("Model loading time: {}", elapsed_time.as_secs_f64());

    /* ARCHITECTURE BUILDING */
    parser.borrow_mut().set_next_block(hash_table.clone());
    hash_table.borrow_mut().set_next_block(flow_manager.clone());
    hash_table.borrow_mut().set_discarded_next_block(cms.clone());
    flow_manager.borrow_mut().set_model_next_block(model_wrapper.clone());
    flow_manager.borrow_mut().set_evicted_next_block(cms.clone());
    model_wrapper.borrow_mut().set_mice_next_block(cms.clone());
    model_wrapper.borrow_mut().set_hh_next_block(hash_table.clone());
    cms.borrow_mut().set_control_plane_block(control_plane.clone());

    // compute simulation time 
    let now = Instant::now();

    /* COMMON CODE */
    let mut count = 0;
    parser.borrow_mut().parse_pcap();
    let mut pkt_list = parser.borrow().get_packet_list();
    let mut true_stats_per_epoch = HashMap::new();
    for pkt in &mut pkt_list {
        count += 1;
        //eprintln!("pkt #{} timestamp {}", count, first_pkt.timestamp);
        if count % 100000 == 0 {
            println!("Packet {}", count);
            println!("Time from start {}", (Instant::now() - now).as_secs_f64());
        }
        let mut nb = parser.borrow_mut().next_block.clone().unwrap();

        // true stats per epoch
        if let Some(v) = true_stats_per_epoch.get_mut(&pkt.get_binary_key()) {
            *v += 1;
        } else {
            true_stats_per_epoch.insert(pkt.get_binary_key(), 1);
        }

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
    std::fs::create_dir_all(format!("{}/ht_discarded", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/fm", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/fm_evicted", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/fm_collected", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/model_predictions", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/control_plane", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/simulation_configuration", args.output_dir)).unwrap();

    let mut true_stats_write_file = File::create(format!("{}/gt/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let hash_table_write_file = File::create(format!("{}/ht/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let hash_table_discarded_write_file = File::create(format!("{}/ht_discarded/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let cms_write_file = File::create(format!("{}/cms/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let flow_manager_write_file = File::create(format!("{}/fm/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let mut pkt_cache_received_flows_while_in_cache_evicted_file = File::create(format!("{}/fm_evicted/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let mut pkt_cache_received_flows_while_in_cache_collected_file = File::create(format!("{}/fm_collected/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let mut model_predictions_file = File::create(format!("{}/model_predictions/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let mut control_plane_file = File::create(format!("{}/control_plane/{}.csv", args.output_dir, args.output_file_name)).unwrap();
    let simulation_configuration_file = File::create(format!("{}/simulation_configuration/{}.csv", args.output_dir, args.output_file_name)).unwrap();

    // disable some output
    control_plane_file = File::create("/dev/null").unwrap();
    model_predictions_file = File::create("/dev/null").unwrap();
    pkt_cache_received_flows_while_in_cache_collected_file = File::create("/dev/null").unwrap();
    pkt_cache_received_flows_while_in_cache_evicted_file = File::create("/dev/null").unwrap();
    true_stats_write_file = File::create("/dev/null").unwrap();

    let mut true_stats_buf_writer = BufWriter::new(true_stats_write_file);
    let mut hash_table_buf_writer = BufWriter::new(hash_table_write_file);
    let mut hash_table_discarded_buf_writer = BufWriter::new(hash_table_discarded_write_file);
    let mut cms_buf_writer = BufWriter::new(cms_write_file);
    let mut flow_manager_buf_writer = BufWriter::new(flow_manager_write_file);
    let mut pkt_cache_received_flows_while_in_cache_evicted_buf_write = BufWriter::new(pkt_cache_received_flows_while_in_cache_evicted_file);
    let mut pkt_cache_received_flows_while_in_cache_collected_buf_write = BufWriter::new(pkt_cache_received_flows_while_in_cache_collected_file);
    let mut model_predictions_buf_write = BufWriter::new(model_predictions_file);
    let mut control_plane_buf_write = BufWriter::new(control_plane_file);
    let mut simulation_configuration_buf_write = BufWriter::new(simulation_configuration_file);

    for (&k,&v) in &true_stats_per_epoch {
        let string_key = Packet::from_binary_key_to_flow_id_string(k);
        true_stats_buf_writer.write_fmt(format_args!("{},{}\n", string_key, v)).unwrap();
    }
    for string in hash_table.borrow().dump_data_formatted() {
        hash_table_buf_writer.write_fmt(format_args!("{}\n", string)).unwrap();
    }
    for k in control_plane.borrow().get_keys() {
        let size = cms.borrow().get_size_estimate(k);
        let string_key = Packet::from_binary_key_to_flow_id_string(k);
        cms_buf_writer.write_fmt(format_args!("{},{}\n", string_key, size)).unwrap();
    }
    for (k,v) in flow_manager.borrow_mut().dump_data() {
        let string_key = Packet::from_binary_key_to_flow_id_string(k);
        flow_manager_buf_writer.write_fmt(format_args!("{},{}\n", string_key, v)).unwrap();
    }

    for v in flow_manager.borrow_mut().get_received_flows_while_in_cache_evicted() {
        pkt_cache_received_flows_while_in_cache_evicted_buf_write.write_fmt(format_args!("{},{},{},{},{}\n",v.0, v.1, v.2, v.3, v.4)).unwrap();
    }

    for v in flow_manager.borrow_mut().get_received_flows_while_in_cache_collected() {
        pkt_cache_received_flows_while_in_cache_collected_buf_write.write_fmt(format_args!("{},{},{},{},{}\n",v.0, v.1, v.2, v.3, v.4)).unwrap();
    }

    for fl in hash_table.borrow().get_discarded_keys_list() {
        hash_table_discarded_buf_writer.write_fmt(format_args!("{}\n", Packet::from_binary_key_to_flow_id_string(fl))).unwrap();
    }

    for v in control_plane.borrow().get_debug_data() {
        control_plane_buf_write.write_fmt(format_args!("{},{}\n", v.0, v.1)).unwrap();
    }

    for (key, proba_vec) in model_wrapper.borrow().dump_debug_inference() {
        model_predictions_buf_write.write_fmt(format_args!("{},", key)).unwrap();
        model_predictions_buf_write.write_fmt(format_args!("[")).unwrap();
        let temp_len = proba_vec.len();
        for (i,v) in proba_vec.iter().enumerate() {
            if i == (temp_len - 1) {
                model_predictions_buf_write.write_fmt(format_args!("{}-{}]\n", v.0, v.1)).unwrap();
            } else {
                model_predictions_buf_write.write_fmt(format_args!("{}-{},", v.0, v.1)).unwrap();
            }
        }
    }

    let elapsed_time = Instant::now() - now;
    print!("Simulation time {}", elapsed_time.as_secs_f64());

    simulation_configuration_buf_write.write_fmt(format_args!("Simulation Time: {} s\n", elapsed_time.as_secs())).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("PCAP file: {}\n", args.pcap_file)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Model file: {}\n", args.model_file)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Model threshold: {}\n", args.model_threshold)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("System memory: {} KB\n", args.flow_manager_slots *args.flow_manager_rows *(6+8+1)/1024 + args.ht_slots*args.ht_rows*args.ht_count*6/1024 + 531)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("FSE memory: {} KB\n", args.ht_slots*args.ht_rows*args.ht_count*3/1024 + args.cms_cols*(3 + (2*(args.cms_rows - 1)))/1024)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Elephant Tracker:\n")).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - tables: {}\n", args.ht_count)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - rows: {}\n", args.ht_rows)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - slots: {}\n", args.ht_slots)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - total entries: {}\n", args.ht_slots*args.ht_rows*args.ht_count)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - memory: {} KB\n", args.ht_slots*args.ht_rows*args.ht_count*6/1024)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - extra memory (FSE): {} KB\n", args.ht_slots*args.ht_rows*args.ht_count*3/1024)).unwrap();
    // simulation_configuration_buf_write.write_fmt(format_args!(" - extra memory (IAT): {} KB\n", args.ht_slots*args.ht_rows*args.ht_count*2/1024)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - load factor: {}\n", hash_table.borrow().get_load_factor())).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Mice Tracker\n")).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - (FSE) CMS cols: {}\n", args.cms_cols)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - (FSE) CMS load factor: {}\n", cms.borrow().get_load_factor())).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - (FSE) CMS memory: {} KB\n", args.cms_cols*(3 + (2*(args.cms_rows - 1)))/1024)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Flow Manager (k={})\n", args.flow_manager_packet_limit)).unwrap();
    match args.flow_manager_type {
        FlowManagerType::Layered => {
            for (i,x) in (flow_manager.borrow().as_any().downcast_ref::<HierarchicalFlowManager>()).unwrap().rows.iter().enumerate() {
                simulation_configuration_buf_write.write_fmt(format_args!(" - layer {} rows: {}\n", i, x)).unwrap();
            }
        }
        _ => {
            simulation_configuration_buf_write.write_fmt(format_args!(" - Flow Manager rows: {}\n", args.flow_manager_rows)).unwrap();
        }
    }
    simulation_configuration_buf_write.write_fmt(format_args!(" - slots: {}\n", args.flow_manager_slots)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - slot size: {}\n", args.flow_manager_slot_size)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!(" - memory: {} KB\n", args.flow_manager_slots *args.flow_manager_rows *args.flow_manager_slot_size /1024)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Control Plane accesses: {}\n", control_plane.borrow().get_transmitted_keys_count())).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Predicted Elephants: {}\n", model_wrapper.borrow_mut().get_inner_struct().inference_count_hh)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Predicted Mice: {}\n", model_wrapper.borrow_mut().get_inner_struct().inference_count_mice)).unwrap();

    true_stats_buf_writer.flush().unwrap();
    hash_table_buf_writer.flush().unwrap();
    hash_table_discarded_buf_writer.flush().unwrap();
    cms_buf_writer.flush().unwrap();
    flow_manager_buf_writer.flush().unwrap();
    pkt_cache_received_flows_while_in_cache_evicted_buf_write.flush().unwrap();
    pkt_cache_received_flows_while_in_cache_collected_buf_write.flush().unwrap();
    model_predictions_buf_write.flush().unwrap();
    simulation_configuration_buf_write.flush().unwrap();
}
