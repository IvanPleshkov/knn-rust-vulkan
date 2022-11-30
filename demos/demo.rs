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
    let vectors_count = 128 * 4;
    let vector_dim = 512;
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
    let result = knn_worker.knn(&query, vectors_count);

    println!("result: {:?}", result);

    for (i, v) in list.iter().enumerate() {
        println!("({}, {})", i, (dot(&query, v) - result[i].1).abs());
    }

    println!("finish query vectors");
}
