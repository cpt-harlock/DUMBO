pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

pub mod bloom_filter;
pub mod cms;
pub mod hash_table;
pub mod model_wrapper;
pub mod packet;
pub mod flow_manager;
pub mod parser;
pub mod control_plane;
pub mod ddsketch;
pub mod pcap;
