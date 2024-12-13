use std::collections::{HashMap, HashSet};

use ndarray::{Data, Array1, Array2, ArrayBase, Ix1, Ix2};

use crate::data::MatrixDataSource;
use crate::heaps::{GenericHeap, MinHeap};
use crate::types::{UnsignedInteger, Float, trait_combiner};
use crate::measures::Distance;
use crate::bit_vectors::{BitVector, BitVectorMut};

pub trait RangeIndex<R: UnsignedInteger, F: Float, Dist: Distance<F>> {
	fn range_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix1>, range: F) -> (Array1<R>, Array1<F>);
	fn range_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix2>, range: F) -> (Vec<Array1<R>>, Vec<Array1<F>>);
}
pub trait KnnIndex<R: UnsignedInteger, F: Float, Dist: Distance<F>> {
	fn knn_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix1>, k: usize) -> (Array1<R>, Array1<F>);
	fn knn_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix2>, k: usize) -> (Array2<R>, Array2<F>);
}
trait_combiner!(GeneralIndex[R: UnsignedInteger, F: Float, Dist: (Distance<F>)]: (RangeIndex<R, F, Dist>) + (KnnIndex<R, F, Dist>) + (MatrixDataSource<F>));


pub trait Graph<R: UnsignedInteger> {
	fn n_vertices(&self) -> usize;
	fn n_edges(&self) -> usize;
	fn neighbors(&self, vertex: R) -> Vec<R>;
	fn ego_graph_nodes_hashset(&self, vertex: R, radius: usize) -> Vec<R> {
		let mut visited = HashSet::new();
		let mut work = vec![vertex];
		let mut visited_ids = vec![vertex];
		visited.insert(vertex);
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				for &n in &self.neighbors(v) {
					if !visited.contains(&n) {
						visited.insert(n);
						next_work.push(n);
						visited_ids.push(n);
					}
				}
			}
			work = next_work;
		}
		visited_ids
	}
	fn ego_graph_nodes_bitvec(&self, vertex: R, radius: usize) -> Vec<R> {
		let mut visited = vec![0u64; (self.n_vertices()+63)/64];
		let mut work = vec![vertex];
		let mut visited_ids = vec![vertex];
		visited.set_bit_unchecked(vertex.to_usize().unwrap(), true);
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				for &n in &self.neighbors(v) {
					if !visited.get_bit_unchecked(n.to_usize().unwrap()) {
						visited.set_bit_unchecked(n.to_usize().unwrap(), true);
						next_work.push(n);
						visited_ids.push(n);
					}
				}
			}
			work = next_work;
		}
		visited_ids
	}
}
pub trait WeightedGraph<R: UnsignedInteger, F: Float>: Graph<R> {
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F;
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<R>, Vec<F>);
	fn weighted_ego_graph_nodes(&self, vertex: R, radius: usize) -> (Vec<R>, Vec<F>) {
		let mut visited = HashMap::new();
		let mut work = vec![vertex];
		visited.insert(vertex, F::zero());
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				let v_dist = visited[&v];
				let (neighbors, weights) = self.neighbors_with_weights(v);
				for (&n, &w) in neighbors.iter().zip(weights.iter()) {
					if !visited.contains_key(&n) || visited[&n] > v_dist + w {
						visited.insert(n, v_dist + w);
						next_work.push(n);
					}
				}
			}
			work = next_work;
		}
		let mut heap = MinHeap::new();
		for (&v, &d) in &visited {
			heap.push(d, v);
		}
		let mut visited_ids = Vec::with_capacity(heap.size());
		let mut visited_dists = Vec::with_capacity(heap.size());
		for (d, v) in heap {
			visited_ids.push(v);
			visited_dists.push(d);
		}
		(visited_ids, visited_dists)
	}
}


#[derive(Debug, Clone)]
pub struct NoSuchLayerError;
pub trait GraphIndex<R: UnsignedInteger, F: Float, Dist: Distance<F>>: GeneralIndex<R, F, Dist> {
	/// Returns the number of layers in the graph index.
	fn layer_count(&self) -> usize;
	/// Returns the graph at the given layer if available, otherwise returns an error.
	/// 0 is the lowest level, layer_count()-1 is the highest level.
	fn get_layer(&self, layer: usize) -> Result<&dyn Graph<R>, NoSuchLayerError>;
	/// Executes a greedy search on the given layer with a maximum heap size and returns the heap containing the results.
	fn greedy_search_layer<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, max_heap_size: usize, layer: usize) -> MinHeap<F,R> {
		self.greedy_search_layer_with_heap(q, MinHeap::new(), max_heap_size, layer)
	}
	/// Executes a greedy search on the hierarchy with a maximum heap size and returns the heap containing the results.
	fn greedy_search<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, max_heap_size: usize) -> MinHeap<F,R> {
		self.greedy_search_with_heap(q, MinHeap::new(), max_heap_size)
	}
	/// Executes a greedy search on the given layer with a potentially pre-filled heap and returns the heap containing the results.
	fn greedy_search_layer_with_heap<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, heap: MinHeap<F,R>, max_heap_size: usize, layer: usize) -> MinHeap<F,R>;
	/// Executes a greedy search on the hierarchy with a potentially pre-filled heap and returns the heap containing the results.
	fn greedy_search_with_heap<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, mut heap: MinHeap<F,R>, max_heap_size: usize) -> MinHeap<F,R> {
		for layer in (0..self.layer_count()).rev() {
			heap = self.greedy_search_layer_with_heap(q, heap, max_heap_size, layer);
		}
		heap
	}
}

