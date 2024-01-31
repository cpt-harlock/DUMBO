#[derive(Debug)]
pub struct LogMaxMinMapping {
    pub accuracy: f64,
    pub gamma: f64,
    pub max: f64,
    pub min: f64,
    pub max_key: i32,
    pub min_key: i32,
    pub bins_count: usize,
}

impl LogMaxMinMapping {
    pub fn build_mapping(max: f64, min: f64, bins_count: usize) -> LogMaxMinMapping {
        let mut accuracy: f64;
        let mut gamma: f64;
        let mut max_key: u32;
        let mut min_key: u32;
        let bins: Vec<u32> = vec![0; bins_count as usize];
        let mut temp_gamma = (max / min).powf(1.0 / ((bins_count - 1) as f64));
        let mut temp_max_key = ((max.log2() / temp_gamma.log2()).ceil()) as i32;
        let mut temp_min_key = ((min.log2() / temp_gamma.log2()).ceil()) as i32;
        let mut computed_bins = ((temp_max_key - temp_min_key) + 1);
        let mut old_direction = computed_bins > (bins_count as i32);
        let mut new_direction: bool;
        let mut max_iterations = 100;
        let mut factor = 0.01;
        let gamma = loop {
            if max_iterations == 0 {
                println!(
                    "{},{},{},{},{}",
                    max, min, temp_max_key, temp_min_key, temp_gamma
                );
                panic!("Error");
            }
            if computed_bins as usize == bins_count {
                break temp_gamma;
            } else if computed_bins as usize > bins_count {
                temp_gamma *= (1.0 + factor);
            } else {
                temp_gamma *= (1.0 - factor);
            }
            temp_max_key = ((max.log2() / temp_gamma.log2()).ceil()) as i32;
            temp_min_key = ((min.log2() / temp_gamma.log2()).ceil()) as i32;
            computed_bins = ((temp_max_key - temp_min_key) + 1);
            new_direction = computed_bins > (bins_count as i32);
            max_iterations -= 1;
            if (new_direction != old_direction) {
                factor /= 2.0;
            }
            old_direction = new_direction;
            println!("max iter {} gamma {}", max_iterations, temp_gamma);
        };
        accuracy = ((2.0 * gamma) / (gamma + 1.0)) - 1.0;
        //println!("accuracy {}", accuracy);
        let max_key = ((max.log2() / gamma.log2()).ceil()) as i32;
        let min_key = ((min.log2() / gamma.log2()).ceil()) as i32;
        let s = LogMaxMinMapping {
            accuracy,
            gamma,
            max,
            min,
            max_key,
            min_key,
            bins_count,
        };
        s
    }
    /// Map a value to a key
    pub fn val_to_key(&self, val: f64) -> i32 {
        let k = self.log_gamma(val);
        if (k > self.max_key) {
            return self.max_key;
        } else if (k < self.min_key) {
            return self.min_key;
        } else {
            return k;
        }
    }

    /// Map a value to a bin
    pub fn val_to_bin(&self, val: f64) -> usize {
        let k = self.val_to_key(val);
        (k - self.min_key) as usize
    }

    pub fn bin_upper_delimiter(&self, bin_index: i32) -> f64 {
        self.gamma.powi(bin_index+self.min_key)
    }

    pub fn bin_lower_delimiter(&self, bin_index: i32) -> f64 {
        self.gamma.powi(bin_index+self.min_key-1)
    }

    /// Key to val approximation
    pub fn key_to_val_approximation(&self, key: i32) -> f64 {
        //println!("gamma {}", self.gamma);
        //println!("key {}", key);
        //println!("min key {}", self.min_key);
        //println!("max key {}", self.max_key);
        (2.0 * self.gamma.powi(key)) / (self.gamma + 1.0)
    }
    /// Computing log base gamma of a value
    fn log_gamma(&self, val: f64) -> i32 {
        ((val.log2() / self.gamma.log2()).ceil()) as i32
    }
}
