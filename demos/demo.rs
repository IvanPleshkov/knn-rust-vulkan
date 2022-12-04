use rand::Rng;
use std::{
    collections::{BinaryHeap, LinkedList},
    sync::Arc,
};

use knn_rust_vulkan::{
    debug_messenger::PanicIfErrorMessenger, gpu_device::GpuDevice, gpu_instance::GpuInstance,
    knn_worker::Score,
};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut result = 0.0;
    for i in 0..a.len() {
        result += a[i] * b[i];
    }
    result
}

fn main() {
    let vectors_count = 1024;
    //let vectors_count = 64 * 63;
    let vector_dim = 64;
    let k = 5;

    let debug_messenger = PanicIfErrorMessenger {};
    let instance = Arc::new(GpuInstance::new("KNN vulkan", Some(&debug_messenger), false).unwrap());
    let device =
        Arc::new(GpuDevice::new(instance.clone(), instance.vk_physical_devices[0]).unwrap());
    let mut knn_worker =
        knn_rust_vulkan::knn_worker::KnnWorker::new(device.clone(), vector_dim, vectors_count);

    println!("Generate data");
    let mut rng = rand::thread_rng();
    let mut list: LinkedList<Vec<f32>> = LinkedList::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        list.push_back(vector);
    }

    println!("start adding vectors");
    let timer = std::time::Instant::now();
    for (i, v) in list.iter().enumerate() {
        knn_worker.add_vector(v, i);
    }
    knn_worker.flush();
    println!("uploading vectors time {:?}", timer.elapsed());
    println!("finish adding vectors");

    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();

    println!("start searching gpu");
    let timer = std::time::Instant::now();
    let result = knn_worker.knn(&query, k);
    println!("finish searching gpu in {:?}", timer.elapsed());
    println!("result: {:?}", result);

    println!("start searching cpu");
    let timer = std::time::Instant::now();
    let scores: Vec<Score> = {
        let mut heap: BinaryHeap<Score> = BinaryHeap::new();
        for (i, v) in list.iter().enumerate() {
            let score = Score {
                score: dot(&query, &v),
                index: i as u32,
            };
            if heap.len() == k {
                let mut top = heap.peek_mut().unwrap();
                if top.score > score.score {
                    *top = score;
                }
            } else {
                heap.push(score);
            }
        }
        heap.into_sorted_vec()
    };
    println!("finish searching cpu in {:?}", timer.elapsed());
    println!("orig: {:?}", scores);

    let idx1 = result.iter().map(|x| x.index).collect::<Vec<_>>();
    let idx2 = scores.iter().map(|x| x.index).collect::<Vec<_>>();
    assert_eq!(idx1, idx2);

    println!("finish query vectors");
}
