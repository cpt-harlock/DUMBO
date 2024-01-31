#![feature(trait_upcasting)]    // TODO remove this once feature is merged in stable (https://github.com/rust-lang/rust/pull/118133)

use CODA_simu::ddsketch::{*};
use CODA_simu::parser::Parser;
use CODA_simu::packet::*;
use CODA_simu::flow_manager::*;
use CODA_simu::model_wrapper::*;
use clap::Parser as ClapParser;
use std::rc::Rc;
use std::cell::RefCell;
use std::time::Instant;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
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
    Hierarchical
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
    #[clap(value_enum)]
    flow_manager_type: FlowManagerType,
    flow_manager_rows: usize,
    flow_manager_slots: usize,
    flow_manager_slot_size: usize,
    flow_manager_packet_limit: usize,
    flow_manager_hierarchy: String,
    bf_size: usize,
    bf_hash_count: usize,
    ddsketch_hh_max: f64,
    ddsketch_hh_min: f64,
    max_hh: usize,
    model_threshold: f64,
    model_ap: f64,
    model_emr: f64,
    model_mmr: f64,
    #[clap(short, long, action)]
    tcp_only: bool,
    #[clap(short, long, action)]
    run_baselines: bool,
}

fn main() {
    let now;
    let elapsed_time;
    let args = Args::parse();
    let parser = Rc::new(RefCell::new(Parser::build_parser(&args.pcap_file, args.tcp_only)));
    let hh_ddsketch_array = Rc::new(RefCell::new(DDSketchArray::build_ddsketch_array(args.ddsketch_hh_max, args.ddsketch_hh_min, 32,16)));
    let ddsketch_filter = Rc::new(RefCell::new(DDSketchFilter::build_ddsketch_filter(args.max_hh)));
    //let temp_ddsketch = DDSketch::build_ddsketch(args.ddsketch_hh_max, args.ddsketch_hh_min, args.ddsketch_hh_bins, args.ddsketch_hh_max, args.ddsketch_hh_min, 0, 0, 32);
    // get the value corresponding to the number of bins for mice
    //let mice_max = temp_ddsketch.central_mapping.bin_lower_delimiter((args.ddsketch_mice_bins) as i32);
    //let mice_ddsketch = DDSketch::build_ddsketch(mice_max, args.ddsketch_hh_min, args.ddsketch_mice_bins, mice_max, args.ddsketch_hh_min, 0, 0, 32);
    let mice_ddsketch_array = Rc::new(RefCell::new(DDSketchArray::build_ddsketch_array(args.ddsketch_hh_max, args.ddsketch_hh_min, 31, 8)));
    let quantiles = vec![0.5, 0.75, 0.9, 0.95, 0.99];

    // Flow Manager (former packet cache)
    let flow_manager: Rc<RefCell<dyn FlowManager>> = match args.flow_manager_type {
        FlowManagerType::EvictOldest => { Rc::new(RefCell::new(FlowManagerEvictOldest::build(args.flow_manager_rows, args.flow_manager_slots, args.flow_manager_packet_limit, 323))) }
        FlowManagerType::EvictTimeout => { Rc::new(RefCell::new(FlowManagerEvictTimeout::build(args.flow_manager_rows, args.flow_manager_slots, args.flow_manager_packet_limit, 323, 2000.0))) }
        FlowManagerType::Infinite => { Rc::new(RefCell::new(InfiniteFlowManager::build(args.flow_manager_packet_limit))) }
        FlowManagerType::Hierarchical => {
            //let proportions = vec![1.0, 0.85, 0.7, 0.62];
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
    parser.borrow_mut().set_next_block(ddsketch_filter.clone());
    ddsketch_filter.borrow_mut().set_flow_manager(flow_manager.clone());
    ddsketch_filter.borrow_mut().set_hh_ddsketch(hh_ddsketch_array.clone());
    ddsketch_filter.borrow_mut().set_mice_ddsketch(mice_ddsketch_array.clone());
    flow_manager.borrow_mut().set_evicted_next_block(ddsketch_filter.clone());
    model_wrapper.borrow_mut().set_hh_next_block(ddsketch_filter.clone());
    model_wrapper.borrow_mut().set_mice_next_block(ddsketch_filter.clone());
    flow_manager.borrow_mut().set_model_next_block(model_wrapper.clone());

    // compute simulation time 
    let now = Instant::now();

    /* COMMON CODE */
    let mut count = 0;
    parser.borrow_mut().parse_pcap();
    let mut pkt_list = parser.borrow().get_packet_list();
    // clone packet list for the baseline
    let mut pkt_list_clone = pkt_list.to_vec();
    //println!("packet list length: {}", pkt_list.len());
    let mut iat_stats_per_epoch: HashMap<u128,Vec<f64>> = HashMap::new();
    let mut timestamp_stats_per_epoch: HashMap<u128,f64> = HashMap::new();
    for pkt in &mut pkt_list {
        //let mut packet_copy = pkt.clone();
        count += 1;
        //eprintln!("pkt #{} ", count);
        if count % 100000 == 0 {
            println!("Packet {}", count);
            println!("Time from start {}", (Instant::now() - now).as_secs_f64());
        }
        let mut nb = parser.borrow_mut().next_block.clone().unwrap();

        // true stats per epoch
        // already seen a packet for the flow
        if let Some(v) = timestamp_stats_per_epoch.get_mut(&pkt.get_binary_key()) {
            iat_stats_per_epoch.get_mut(&pkt.get_binary_key()).unwrap().push(pkt.timestamp - *v);
            *v = pkt.timestamp;
        } else {
            timestamp_stats_per_epoch.insert(pkt.get_binary_key(), pkt.timestamp);
            iat_stats_per_epoch.insert(pkt.get_binary_key(), vec![]);
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

    /* Baselines */
    // DDSketch with ~ memory -> 1 byte per buckets
    let one_byte_per_bucket_ddsketch_array = Rc::new(RefCell::new(DDSketchArray::build_ddsketch_array(args.ddsketch_hh_max, args.ddsketch_hh_min, 32, 8)));
    // DDSketch with ~ memory -> 2 byte per buckets (half #buckets)
    let two_bytes_per_bucket_ddsketch_array = Rc::new(RefCell::new(DDSketchArray::build_ddsketch_array(args.ddsketch_hh_max, args.ddsketch_hh_min, 16, 16)));
    // DDSketch with ~ x2 memory -> 2 byte per buckets
    let doubled_ddsketch_array = Rc::new(RefCell::new(DDSketchArray::build_ddsketch_array(args.ddsketch_hh_max, args.ddsketch_hh_min, 32, 16)));
    if args.run_baselines {
        count = 0;
        println!("start baseline simulation");
        for pkt in &mut pkt_list_clone {
            count += 1;
            //eprintln!("pkt #{} ", count);
            if count % 100000 == 0 {
                println!("Packet {}", count);
                println!("Time from start {}", (Instant::now() - now).as_secs_f64());
            }
            pkt.agg_features.insert(String::from("timestamp_0"), pkt.timestamp);
            one_byte_per_bucket_ddsketch_array.borrow_mut().receive_packet(&mut pkt.clone());
            two_bytes_per_bucket_ddsketch_array.borrow_mut().receive_packet(&mut pkt.clone());
            doubled_ddsketch_array.borrow_mut().receive_packet(&mut pkt.clone());
            //reduced_ddsketch_array.borrow_mut().receive_packet(&mut second_copy);
        }

        println!("end of simulation");
    }

    /* stats */
    // end of simulation dump
    std::fs::create_dir_all(format!("{}/hh_ddsketch", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/mice_ddsketch", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/flow_manager", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/gt", args.output_dir)).unwrap();
    std::fs::create_dir_all(format!("{}/simulation_configuration", args.output_dir)).unwrap();

    for q in quantiles {
        let output_file_name = format!("{}_{}",args.output_file_name, q);
        let output_baseline_file_name = format!("baseline_{}", q);
        let hh_ddsketch_file = File::create(format!("{}/hh_ddsketch/{}.csv", args.output_dir, output_file_name)).unwrap();
        let mice_ddsketch_file = File::create(format!("{}/mice_ddsketch/{}.csv", args.output_dir, output_file_name)).unwrap();
        let flow_manager_file = File::create(format!("{}/flow_manager/{}.csv", args.output_dir, output_file_name)).unwrap();
        let gt_file = File::create(format!("{}/gt/{}.csv", args.output_dir, output_file_name)).unwrap();

        let mut hh_ddsketch_writer = BufWriter::new(hh_ddsketch_file);
        let mut mice_ddsketch_writer = BufWriter::new(mice_ddsketch_file);
        let mut flow_manager_writer = BufWriter::new(flow_manager_file);
        let mut gt_writer = BufWriter::new(gt_file);
        // hh ddsketch
        for ddsketch in &hh_ddsketch_array.borrow().ddsketch_array {
            let quantile = ddsketch.1.ddsketch.get_quantile(q);
            let string_key = Packet::from_binary_key_to_flow_id_string(*ddsketch.0);
            hh_ddsketch_writer.write_fmt(format_args!("{},{}\n", string_key, quantile)).unwrap();
        }
        // mice ddsketch
        for ddsketch in &mice_ddsketch_array.borrow().ddsketch_array {
            let quantile = ddsketch.1.ddsketch.get_quantile(q);
            let string_key = Packet::from_binary_key_to_flow_id_string(*ddsketch.0);
            mice_ddsketch_writer.write_fmt(format_args!("{},{}\n", string_key, quantile)).unwrap();
        }
        // flow manager
        for (key, packet_vector) in flow_manager.borrow_mut().dump_raw_data() {
            let string_key = Packet::from_binary_key_to_flow_id_string(key);
            if packet_vector.len() == 1 {
                flow_manager_writer.write_fmt(format_args!("{},{}\n", string_key, 0)).unwrap();
            } else {
                let mut iat_vector = vec![];
                let mut temp = 0.0;
                for (i,tmsp) in packet_vector.iter().enumerate() {
                    if i == 0 {
                        temp = tmsp.timestamp;
                        continue;
                    }
                    let iat = tmsp.timestamp - temp;
                    iat_vector.push(iat);
                    temp = tmsp.timestamp;
                }
                let vec_len = iat_vector.len();
                let quantile_index = ((q*(vec_len as f64 - 1.0))+1.0).floor() as usize;
                iat_vector.sort_by(|a, b| a.partial_cmp(b).unwrap());
                //println!("vec len: {}", vec_len);
                //println!("quantile index: {}", quantile_index);
                let val = iat_vector[quantile_index-1];
                flow_manager_writer.write_fmt(format_args!("{},{}\n", string_key, val)).unwrap();
            }
        }
        if args.run_baselines {
            std::fs::create_dir_all(format!("{}/one_byte_per_bucket_ddsketch", args.output_dir)).unwrap();
            std::fs::create_dir_all(format!("{}/two_bytes_per_bucket_ddsketch", args.output_dir)).unwrap();
            std::fs::create_dir_all(format!("{}/doubled_ddsketch", args.output_dir)).unwrap();

            let one_byte_per_bucket_ddsketch_file = File::create(format!("{}/one_byte_per_bucket_ddsketch/{}.csv", args.output_dir, output_baseline_file_name)).unwrap();
            let two_bytes_per_bucket_ddsketch_file = File::create(format!("{}/two_bytes_per_bucket_ddsketch/{}.csv", args.output_dir, output_baseline_file_name)).unwrap();
            let doubled_ddsketch_file = File::create(format!("{}/doubled_ddsketch/{}.csv", args.output_dir, output_baseline_file_name)).unwrap();

            let mut one_byte_per_bucket_ddsketch_writer = BufWriter::new(one_byte_per_bucket_ddsketch_file);
            let mut two_bytes_per_bucket_ddsketch_writer = BufWriter::new(two_bytes_per_bucket_ddsketch_file);
            let mut doubled_ddsketch_writer = BufWriter::new(doubled_ddsketch_file);

            // one byte ddsketch
            for ddsketch in &one_byte_per_bucket_ddsketch_array.borrow().ddsketch_array {
                let quantile = ddsketch.1.ddsketch.get_quantile(q);
                let string_key = Packet::from_binary_key_to_flow_id_string(*ddsketch.0);
                one_byte_per_bucket_ddsketch_writer.write_fmt(format_args!("{},{}\n", string_key, quantile)).unwrap();
            }
            // two bytes ddsketch
            for ddsketch in &two_bytes_per_bucket_ddsketch_array.borrow().ddsketch_array {
                let quantile = ddsketch.1.ddsketch.get_quantile(q);
                let string_key = Packet::from_binary_key_to_flow_id_string(*ddsketch.0);
                two_bytes_per_bucket_ddsketch_writer.write_fmt(format_args!("{},{}\n", string_key, quantile)).unwrap();
            }
            // doubled ddsketch
            for ddsketch in &doubled_ddsketch_array.borrow().ddsketch_array {
                let quantile = ddsketch.1.ddsketch.get_quantile(q);
                let string_key = Packet::from_binary_key_to_flow_id_string(*ddsketch.0);
                doubled_ddsketch_writer.write_fmt(format_args!("{},{}\n", string_key, quantile)).unwrap();
            }
        }
        //true stats 
        for (k,mut vector) in iat_stats_per_epoch.clone() {
            let string_key = Packet::from_binary_key_to_flow_id_string(k);
            let vec_len = vector.len();
            if vec_len == 0 {
                gt_writer.write_fmt(format_args!("{},{}\n", string_key, 0)).unwrap();
                continue;
            }
            let quantile_index = ((q*(vec_len as f64 - 1.0))+1.0).floor() as usize;
            vector.sort_by(|a, b| a.partial_cmp(b).unwrap());
            //println!("vec len: {}", vec_len);
            //println!("quantile index: {}", quantile_index);
            let val = vector[quantile_index-1];
            gt_writer.write_fmt(format_args!("{},{}\n", string_key, val)).unwrap();
        }
        if args.run_baselines {
            gt_writer.flush().unwrap();
            std::fs::copy(format!("{}/gt/{}.csv", args.output_dir, output_file_name), format!("{}/gt/{}.csv", args.output_dir, output_baseline_file_name)).expect("TODO: panic message");
        }
    }

    let simulation_configuration_file = File::create(format!("{}/simulation_configuration/{}.csv", args.output_dir, args.output_file_name)).unwrap();

    let mut simulation_configuration_buf_write = BufWriter::new(simulation_configuration_file);

    simulation_configuration_buf_write.write_fmt(format_args!("HH DDSketch bin: {}\n", 32)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("HH DDSketch bin size: {}\n", 2)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Mice DDSketch bin: {}\n", 31)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Mice DDSketch bin size: {}\n", 1)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Predicted Elephants: {}\n", model_wrapper.borrow_mut().get_inner_struct().inference_count_hh)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Predicted Mice: {}\n", model_wrapper.borrow_mut().get_inner_struct().inference_count_mice)).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Elephant Tracker: {}\n", hh_ddsketch_array.borrow().ddsketch_array.len())).unwrap();
    simulation_configuration_buf_write.write_fmt(format_args!("Mice Tracker: {}\n", mice_ddsketch_array.borrow().ddsketch_array.len())).unwrap();
}
