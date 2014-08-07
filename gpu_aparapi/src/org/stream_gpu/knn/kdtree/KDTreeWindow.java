package org.stream_gpu.knn.kdtree;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Queue;

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
	
	public KDTreeWindow (int window_size,Instances dataset)
	{
		m_current_id = 0;
		m_items = new ArrayDeque<TreeItem>(window_size);
		m_window_size = window_size;
		m_root = new KDTreeNode(dataset, null);
	}
	
	public void add( Instance inst)
	{
		++m_current_id;
		if (m_current_id == Long.MAX_VALUE)
			m_current_id = Long.MIN_VALUE;
		
		TreeItem to_add = new TreeItem(m_current_id, inst);
		TreeItem to_remove = null;
		if (m_items.size() >= m_window_size)
		{
			to_remove = m_items.remove();
		}
		m_items.add(to_add);
		m_root.update(to_add,to_remove);
	}
	
	public ArrayList<KDTreeNode> findNearest(Instance test)
	{
		return m_root.findNearest(test);
	}


	public void print() {
		m_root.print(System.out, 0);
	}

	public int size() {
		return m_root.size();
	}
	
	
}
