package org.stream_gpu.knn.kdtree;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Iterator;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

public class KDTreeNode {

	public static final int SPLIT_VALUE = 4;
	public static final int COLLAPSE_VALUE = 2;

	private class Range {
		public double m_max;
		public double m_min;

		public Range() {
			m_min = Double.MAX_VALUE;
			m_max = -Double.MAX_VALUE;
		}
	}

	private Instances m_dataset;

	private boolean m_is_leaf;

	private KDTreeNode m_left = null;

	private double m_max_distance;

	private KDTreeNode m_parent;

	private Range[] m_ranges;

	private KDTreeNode m_right = null;

	private int m_split_index;

	private double m_split_value;

	private ArrayList<TreeItem> m_values;

	private int m_window_size;

	public KDTreeNode(Instances dataset, KDTreeNode parent) {
		m_parent = parent;
		m_is_leaf = true;
		m_dataset = dataset;
		m_values = new ArrayList<TreeItem>();
		clearRanges();
	}

	public KDTreeNode(Instances dataset, KDTreeNode parent, ArrayList<TreeItem> values) {
		m_parent = parent;
		m_is_leaf = true;
		m_dataset = dataset;
		m_values = values;
		rescanRanges();
		if (shouldSplit())
			split();
	}

	private void clearRanges() {
		m_split_index = -1;
		m_split_value = Double.MAX_VALUE;
		m_max_distance = Double.MIN_VALUE;
		m_ranges = new Range[m_dataset.numAttributes()];
		for (int i = 0; i < m_ranges.length; i++)
			m_ranges[i] = new Range();
	}

	private void add(TreeItem to_add) {
		if (m_is_leaf)
			insert(to_add);
		else {
			double value = to_add.instance().value(m_split_index);
			if (value <= m_split_value)
				m_left.add(to_add);
			else
				m_right.add(to_add);
		}

	}

	public ArrayList<TreeItem> instances() {
		if (isLeaf())
			return m_values;
		ArrayList<TreeItem> ret = new ArrayList<TreeItem>();
		ret.addAll(m_left.instances());
		ret.addAll(m_right.instances());
		return ret;
	}

	public void insert(TreeItem inst) {
		m_values.add(inst);
		updateRanges(inst);
		if (shouldSplit())
			split();
	}

	public boolean isLeaf() {
		return m_is_leaf;
	}

	private void remove(TreeItem to_remove) {
		if (isLeaf())
		{
			if (!m_values.remove(to_remove))
			{
				System.out.println("Failed to remove "+ to_remove.id());
			}
			rescanRanges();
		}
		else
		{
			if (to_remove.instance().value(m_split_index) <= m_split_value)
				m_left.remove(to_remove);
			else
				m_right.remove(to_remove);
		}
	}
	
	private void rescanRanges()
	{
		clearRanges();
		Iterator<TreeItem> it = m_values.iterator();
		while (it.hasNext())
			updateRanges(it.next());
		
	}
	

	private boolean shouldSplit() {
		return m_values.size() > SPLIT_VALUE;
	}

	public int size() {
		if (isLeaf())
			return m_values.size();
		else
			return m_left.size() + m_right.size();// fix recursion
	}

	private void split() {
		int dim = m_split_index;
		double split = m_split_value;
		
		updateRanges();
		
		assert( dim == m_split_index);
		assert( split == m_split_value);
		
		ArrayList<TreeItem> left_instances = new ArrayList<TreeItem>();
		ArrayList<TreeItem> right_instances = new ArrayList<TreeItem>();
		Iterator<TreeItem> it = m_values.iterator();
		while (it.hasNext())
		{
			TreeItem inst = it.next();
			if (inst.instance().value(dim) <= split)
				left_instances.add(inst);
			else if (inst.instance().value(dim) > split)
				right_instances.add(inst);
		}
		KDTreeNode left = new KDTreeNode(m_dataset, this,left_instances);
		KDTreeNode right = new KDTreeNode(m_dataset, this,right_instances);
		m_left = left;
		if (left.size() == 0)
			throw new RuntimeException("Empty left leaf");
		m_right = right;
		if (right.size() == 0)
			throw new RuntimeException("Empty right leaf");
		m_values.clear();
		m_is_leaf = false;
	}

	private void updateRanges() {
		clearRanges();
		for (int i = 0 ; i < m_values.size(); i ++)
		{
			updateRanges( m_values.get(i) );
		}
	}

	public void update(TreeItem to_add, TreeItem to_remove) {
		int size1 = size();	
		if (to_remove != null)
			remove(to_remove);
		int size2 = size();
		if (to_add != null)
			add(to_add);
		int size3 = size();
		if (size1 == 16 && (size1 != size2+1  || size2 + 1 != size3))
		{
			System.out.println("fail");
			print(System.out, 0);
			remove(to_remove);
		}
	}

	private void updateRanges(TreeItem inst) {
		for (int i = 0; i < inst.instance().numAttributes(); i++) {
			double value = inst.instance().value(i);
			boolean updated = false;
			if (m_ranges[i].m_max < value) {
				m_ranges[i].m_max = value;
				updated = true;
			}
			if (m_ranges[i].m_min > value) {
				m_ranges[i].m_min = value;
				updated = true;
			}
			if (updated) {
				double dist = Math.abs(m_ranges[i].m_max - m_ranges[i].m_min);
				if (dist > m_max_distance && dist > 0) {
					m_max_distance = dist;
					m_split_index = i;
					m_split_value = m_ranges[i].m_min + dist * 0.5;
				}
			}
		}

	}

	public int widestDim() {
		return m_split_index;
	}

	public void print(PrintStream ps, int step) {
		for (int i = 0; i < step; i++)
			ps.print(" ");
		ps.print(m_is_leaf + " split dim " + m_split_index + " value "
				+ m_split_value + " size " + size());
		if (m_is_leaf)
		{
			ps.print(" nodes ");
			Iterator<TreeItem> it = m_values.iterator();
			while (it.hasNext())
				System.out.print(it.next().id() + " ");
			System.out.println();
		}
		else {
			ps.println();
			for (int i = 0; i < step; i++)
				ps.print(" ");

			ps.println("left: ");
			m_left.print(ps, step + 1);
			for (int i = 0; i < step; i++)
				ps.print(" ");
			ps.println("right: ");
			m_right.print(ps, step + 1);
		}
	}

	public void printRanges() {
		System.out.println("Ranges: index " + m_split_index + " value "
				+ m_split_value + " distance " + m_max_distance);
		int i = 0;
		for (Range r : m_ranges) {
			System.out.println((i++) + " " + r.m_min + " <x< " + r.m_max);
		}
	}

	public void findNearest(Instance instance, int k, Heap heap,double distanceToParents) {
		if (isLeaf())
		{
			
		}
		else
		{
		    KDTreeNode nearer, further;
		    boolean targetInLeft = instance.value(m_split_index) <= m_split_value;
  	        if (targetInLeft) {
		        nearer = m_left;
		        further = m_right;
  	        } else {
  	        	nearer = m_right;
			    further = m_left;
  	        }
	/*		
  	        nearer.findNearest(instance, k, heap, distanceToParents);
  	        // ... now look in further half if maxDist reaches into it
			if (heap.size() < k) { // if haven't found the first k
			        double distanceToSplitPlane = distanceToParents
			            + m_EuclidianDistance.sqDifference(node.m_SplitDim, target
			                .value(node.m_SplitDim), node.m_SplitValue);
			        findNearestNeighbours(target, further, k, heap, distanceToSplitPlane);
			        return;
			      } else { // else see if ball centered at query intersects with the other
			                // side.
			        double distanceToSplitPlane = distanceToParents
			            + m_EuclideanDistance.sqDifference(node.m_SplitDim, target
			                .value(node.m_SplitDim), node.m_SplitValue);
			        if (heap.peek().distance >= distanceToSplitPlane) {
			          findNearestNeighbours(target, further, k, heap, distanceToSplitPlane);
			        }
			      }// end else
		}*/
	}
	
	private EuclideanDistance m_EuclidianDistance;
}