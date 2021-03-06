package org.stream_gpu.knn.kdtree;
import java.awt.ItemSelectable;
import java.util.ArrayDeque;
import java.util.ArrayList;






import org.amd.sdk.BitonicSort;
import org.stream_gpu.knn.BitonicSortAparapi;

import com.amd.aparapi.device.Device;

import weka.core.Instance;
import weka.core.Instances;

/** 
 * Implementation of the Sliding Window backed by the KDTree.
 * @author ���
 *
 */
public class KDTreeWindow {
	
	private ArrayDeque<TreeItem> m_items;
	private KDTreeNode m_root;
	private int m_window_size;
	private long m_current_id;
	
	private GpuDistance m_distance;
	private BitonicSort m_sort;
	private GpuInstances m_gpu_model;
	
	public KDTreeWindow (int window_size,Instances dataset)
	{
		m_gpu_model = new GpuInstances(dataset);
		m_distance = new GpuDistance(m_gpu_model);
		m_items = new ArrayDeque<TreeItem>(window_size);
		m_window_size = window_size;
		m_root = new KDTreeNode(dataset, null);
		m_root.SPLIT_VALUE = Math.min(window_size / 16, 32528);
		m_root.COLLAPSE_VALUE = Math.min(window_size / 32, 4096);
		m_distance_kernel = new DistanceKernel(m_gpu_model.length(), m_gpu_model.getNumericsLength());
		m_sort = new BitonicSort();
	}
	
	public void add( Instance inst)
	{
		GpuInstance gpu_instance = m_gpu_model.createInstance(inst);
		
		TreeItem to_add = new TreeItem(m_current_id++, gpu_instance);
		TreeItem to_remove = null;
		if (m_items.size() >= m_window_size)
		{
			to_remove = m_items.remove();
			to_remove.owner().remove( to_remove );
		}
		m_items.add(to_add);
		m_root.add(to_add);
		m_distance = null;
	}
	
	public int size() {
		return m_root.size();
	}
	
	public ArrayList<GpuInstance> findNearest(Instance test, int k)
	{
		if (m_distance ==null) {
			m_distance  = new GpuDistance(m_gpu_model);
			for (TreeItem item : m_items)
				m_distance.update(item.instance());
		}
		Heap h = new Heap(k);
		GpuInstance gpu_instance =m_gpu_model.createInstance(test); 
		findNearest(m_root, gpu_instance, k, h, 0);  
		return h.toArray();
	}
	
	public void findNearest(KDTreeNode node, GpuInstance instance, int k, Heap heap,
			double distanceToParents) {
		if (node.isLeaf()) {
			findNearestForNode(m_gpu_model, heap, instance,  node, k);
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

	protected void findNearestForNode(GpuInstances gpu_model, Heap heap, GpuInstance instance, KDTreeNode node, int k) {
			findNearestForNodeGPU(gpu_model, heap, instance, node, k);
	}
	
	protected void findNearestForNodeCPU(GpuInstances gpu_model, Heap heap, GpuInstance instance, KDTreeNode node) {
		ArrayList<TreeItem> items = node.instances();
		for (TreeItem item : items)
		{
			double distance = m_distance.distance(item.instance(), instance.wekaInstance());
			GpuInstance heapEntry = item.gpuInstance();
			heapEntry.setDistance((float)distance);
			heap.add(heapEntry );
		}
	}
	
	protected void findNearestForNodeGPU(GpuInstances gpu_model, Heap heap, GpuInstance instance, KDTreeNode node, int k) {
		
		float[] test = instance.data();
		ArrayList<TreeItem> items = node.instances();
		float[] values = new float[gpu_model.length() * items.size()];
		int[] indices = new int[items.size()];
		int i = 0;
		long bc  = System.nanoTime();
		for (TreeItem item : items)
		{
			indices[i] = i;
			System.arraycopy(item.gpuInstance().data(), 0, values, (i++) * gpu_model.length() , gpu_model.length());
		}
		
		long bk = System.nanoTime();
		//System.out.println("Prepare:" + (bk - bc ));
		bk = System.nanoTime();
		m_distance_kernel.assign(values,test);
		m_distance_kernel.compute(m_device, k);
		long ac = System.nanoTime();
		//System.out.println("Compute:" + ( ac - bk ));
		ac = System.nanoTime();
		m_sort.sort(Device.firstGPU(), m_distance_kernel.m_results , indices);
		for (i = 0; i < k ; i ++ )
		{
			TreeItem item = items.get( indices[ i ]);
			GpuInstance heapEntry = item.gpuInstance();
			heapEntry.setDistance(m_distance_kernel.m_results[i]);
			heap.add(heapEntry );
		}
		long ah = System.nanoTime();
		//System.out.println("Sort:"+( ah - ac ));
	}
	
	private DistanceKernel m_distance_kernel;
	private Device m_device = Device.firstGPU();
	
	

	public void print() {
		m_root.print(System.out, 0);
	}

}
