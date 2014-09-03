package org.stream_gpu.knn.kdtree;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;

import weka.core.Instance;
import weka.core.Instances;

/** 
 * Implementation of the Sliding Window backed by the KDTree.
 * @author ÐÐÐ
 *
 */
public class KDTreeWindow {
	
	private ArrayDeque<TreeItem> m_items;
	private KDTreeNode m_root;
	private int m_window_size;
	private long m_current_id;
	
	private GpuDistance m_distance;
	private GpuInstances m_gpu_model;
	
	public KDTreeWindow (int window_size,Instances dataset)
	{
		m_gpu_model = new GpuInstances(dataset);
		m_distance = new GpuDistance(m_gpu_model);
		m_items = new ArrayDeque<TreeItem>(window_size);
		m_window_size = window_size;
		m_root = new KDTreeNode(dataset, null);
	}
	
	public void add( Instance inst)
	{
		GpuInstance gpu_instance = m_gpu_model.createInstance(inst);
		
		TreeItem to_add = new TreeItem(m_current_id, gpu_instance);
		TreeItem to_remove = null;
		if (m_items.size() >= m_window_size)
		{
			to_remove = m_items.remove();
			to_remove.owner().remove( to_remove );
		}
		m_items.add(to_add);
		m_root.add(to_add);
		m_distance.update(inst);
	}
	
	public int size() {
		return m_root.size();
	}

	
	public ArrayList<GpuInstance> findNearest(Instance test, int k)
	{
		Heap h = new Heap(k);
		GpuInstance gpu_instance =m_gpu_model.createInstance(test); 
		findNearest(m_root, gpu_instance, k, h, 0);  
		return h.toArray();
	}
	
	
	public void findNearest(KDTreeNode node, GpuInstance instance, int k, Heap heap,
			double distanceToParents) {
		if (node.isLeaf()) {
			findNearestForNode(m_gpu_model, heap, instance,  node);
		} else {
			KDTreeNode nearer, further;
			
			boolean targetInLeft = instance.wekaInstance().value(node.getSplitIndex()) <= node.getSplitValue();
			if (targetInLeft) {
				nearer = node.left();
				further = node.right();
			} else {
				nearer = node.left();
				further = node.right();
			}
			findNearest(nearer,instance, k, heap, distanceToParents);

			if (heap.size() < k) { // if haven't found the first k
				double distanceToSplitPlane = distanceToParents
						+ m_distance.sqDifference(node.getSplitIndex(), 
												 instance.wekaInstance().value(node.getSplitIndex()),
												 node.getSplitValue());
				findNearest(further,instance, k, heap, distanceToSplitPlane);
				return;
			} else {   // else see if ball centered at query intersects with the
						// other
						// side.
				double distanceToSplitPlane = distanceToParents
						+ m_distance.sqDifference(node.getSplitIndex(),
								instance.wekaInstance().value(node.getSplitIndex()),
								node.getSplitValue());
				if (heap.peek().distance() >= distanceToSplitPlane) {
					findNearest(further,instance, k, heap, distanceToSplitPlane);
				}
			}// end else	 
		}
	}

	private void findNearestForNode(GpuInstances gpu_model, Heap heap, GpuInstance instance, KDTreeNode node) {
		ArrayList<TreeItem> items = node.instances();
		for (TreeItem item : items)
		{
			double distance = m_distance.distance(item.instance(), instance.wekaInstance());
			GpuInstance heapEntry = item.gpuInstance();
			heapEntry.setDistance((float)distance);
			heap.add(heapEntry );
		}
	}


	public void print() {
		m_root.print(System.out, 0);
	}

	
	
	
}
