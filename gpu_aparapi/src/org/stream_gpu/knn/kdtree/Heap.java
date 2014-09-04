package org.stream_gpu.knn.kdtree;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;


public class Heap {
	
	
	private PriorityQueue<GpuInstance> m_queue;
	private int m_k;
	
	public Heap(int k)
	{
		m_queue = new PriorityQueue<GpuInstance>(k, new Comparator<GpuInstance>() {
			@Override
			public int compare(GpuInstance left, GpuInstance right) {
				return left.distance() - right.distance() > 0 ? -1 : 1;
			}
		});
		m_k = k;
	}
	
	public void add(GpuInstance inst)
	{
		m_queue.add(inst);
		while (m_queue.size() > m_k)
			m_queue.remove();
	}

	public int size() {
		return m_queue.size();
	}

	public GpuInstance peek() {
		return m_queue.peek();
	}

	public ArrayList<GpuInstance> toArray() {
		ArrayList<GpuInstance> list = new ArrayList<GpuInstance>();
		list.addAll(m_queue);
		return list;
	}

}
