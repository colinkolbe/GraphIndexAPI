use foldhash::HashSet;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{graphs::{Graph, VecViewableAdjGraph, ViewableAdjGraph}, types::SyncUnsignedInteger};

pub fn extend_random_edges<R: SyncUnsignedInteger, T: Graph<R>+Sync+Send>(
	graph: &mut T,
	num_edges: usize,
) {
	let n_vertices = graph.n_vertices();
	let num_threads = rayon::current_num_threads();
	let chunk_size = (n_vertices + num_threads - 1) / num_threads;
	unsafe {
		(0..num_threads).into_par_iter().for_each(|thread_id| {
			let unsafe_graph_ref = graph as *const T as *mut T;
			let mut rng = rand::thread_rng();
			let mut gen_num = move || R::from_usize(rng.gen_range(0..n_vertices)).unwrap_unchecked();
			let start = thread_id * chunk_size;
			let end = (start+chunk_size).min(n_vertices);
			let mut cache_set: HashSet<R> = HashSet::default();
			for i in start..end {
				let i = R::from_usize(i).unwrap_unchecked();
				cache_set.clear();
				cache_set.insert(i);
				match (*unsafe_graph_ref).as_vec_viewable_adj_graph_mut() {
					Some(graph) => {
						let adj = graph.view_neighbors_vec_mut(i);
						adj.iter().cloned().for_each(|v| _=cache_set.insert(v));
						(0..num_edges).for_each(|_| {
							let j = gen_num();
							if cache_set.insert(j) { adj.push(j); }
						});
					}
					None => {
						match graph.as_viewable_adj_graph() {
							Some(graph) => {
								graph.view_neighbors(i).iter().cloned().for_each(|v| _=cache_set.insert(v));
							}
							None => {
								graph.neighbors(i).iter().cloned().for_each(|v| _=cache_set.insert(v));
							}
						}
						(0..num_edges).for_each(|_| {
							let j = gen_num();
							if cache_set.insert(j) { (*unsafe_graph_ref).add_edge(i, j); }
						});
					}
				}
			}
		});
	}
}

pub fn fill_random_edges<R: SyncUnsignedInteger, T: Graph<R>+Sync+Send>(
	graph: &mut T,
	num_edges: usize,
) {
	let n_vertices = graph.n_vertices();
	let num_threads = rayon::current_num_threads();
	let chunk_size = (n_vertices + num_threads - 1) / num_threads;
	unsafe {
		(0..num_threads).into_par_iter().for_each(|thread_id| {
			let unsafe_graph_ref = graph as *const T as *mut T;
			let mut rng = rand::thread_rng();
			let mut gen_num = move || R::from_usize(rng.gen_range(0..n_vertices)).unwrap_unchecked();
			let start = thread_id * chunk_size;
			let end = (start+chunk_size).min(n_vertices);
			let mut cache_set: HashSet<R> = HashSet::default();
			for i in start..end {
				let i = R::from_usize(i).unwrap_unchecked();
				cache_set.clear();
				cache_set.insert(i);
				match (*unsafe_graph_ref).as_vec_viewable_adj_graph_mut() {
					Some(graph) => {
						let adj = graph.view_neighbors_vec_mut(i);
						adj.iter().cloned().for_each(|v| _=cache_set.insert(v));
						let missing = if num_edges+1 >= cache_set.len() { num_edges + 1 - cache_set.len() } else { 0 };
						(0..missing).for_each(|_| {
							let j = gen_num();
							if cache_set.insert(j) { adj.push(j); }
						});
					}
					None => {
						match graph.as_viewable_adj_graph() {
							Some(graph) => {
								graph.view_neighbors(i).iter().cloned().for_each(|v| _=cache_set.insert(v));
							}
							None => {
								graph.neighbors(i).iter().cloned().for_each(|v| _=cache_set.insert(v));
							}
						}
						let missing = if num_edges+1 >= cache_set.len() { num_edges + 1 - cache_set.len() } else { 0 };
						(0..missing).for_each(|_| {
							let j = gen_num();
							if cache_set.insert(j) { (*unsafe_graph_ref).add_edge(i, j); }
						});
					}
				}
			}
		});
	}
}

