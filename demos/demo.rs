use rand::Rng;
use std::{sync::Arc, collections::LinkedList};

use knn_rust_vulkan::{
    debug_messenger::PanicIfErrorMessenger, gpu_device::GpuDevice, gpu_instance::GpuInstance,
};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut result = 0.0;
    for i in 0..a.len() {
        result += a[i] * b[i];
    }
    result
}

fn main() {
    let vectors_count = 128 * 128;
    let vector_dim = 32;
    let k = 16;

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
    for (i, v) in list.iter().enumerate() {
        knn_worker.add_vector(v, i);
    }

    println!("finish adding vectors");

    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
    let result = knn_worker.knn(&query, k);
    println!("result: {:?}", result);

    let mut scores: Vec<(usize, f32)> = list.iter().enumerate().map(|(i, v)| {
        (i, dot(&v, &query))
    }).collect();
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scores = scores.iter().take(k).cloned().collect();
    println!("orig: {:?}", scores);

    let idx1 = result.iter().map(|x| x.0 as usize).collect::<Vec<_>>();
    let idx2 = scores.iter().map(|x| x.0).collect::<Vec<_>>();
    assert_eq!(idx1, idx2);

    println!("finish query vectors");
}
