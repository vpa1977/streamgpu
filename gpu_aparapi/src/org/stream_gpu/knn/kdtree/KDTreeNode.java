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

	public KDTreeNode(Instances dataset, KDTreeNode parent,
			ArrayList<TreeItem> values) {
		m_parent = parent;
		m_is_leaf = true;
		m_dataset = dataset;
		m_values = values;
	
		rescanRanges();
		if (shouldSplit())
			split();
	}

	public void add(TreeItem to_add) {
		if (m_is_leaf) {
			m_values.add(to_add);
			to_add.setOwner(this);
			if (shouldSplit())
				split();
		} else {

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

	public boolean isLeaf() {
		return m_is_leaf;
	}

	public void remove(TreeItem to_remove) {
		if (!isLeaf())
			throw new RuntimeException("Unable to remove from non-leaf node");
		if (!m_values.remove(to_remove))
			throw new RuntimeException("Failed to remove " + to_remove.id());
		if (m_values.size() < COLLAPSE_VALUE && m_parent != null)
			m_parent.collapse(this);
	}
	
	public void collapse( KDTreeNode remove )
	{
		if (m_parent == null)
			return;
		
		ArrayList<TreeItem> items = remove.m_values;
		remove.m_values = null;
		
		KDTreeNode link;
		if (remove == m_left)
			link = m_right;
		else if (remove == m_right)
			link = m_left;
		else
			throw new RuntimeException("Unable to find proper child");
		
		link.m_parent = m_parent;
		if (m_parent.m_left == this)
			m_parent.m_left = link;
		else if (m_parent.m_right == this)
			m_parent.m_right = link;
		else
			throw new RuntimeException("Unable to link to parent");
		
		for (TreeItem item : items)
			m_parent.add(item);
		
		this.m_parent = null;
		this.m_left = null;
		this.m_right = null;
		
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
		rescanRanges();
		if (m_split_index < 0)
			return;
		Iterator<TreeItem> it = m_values.iterator();
		KDTreeNode left = new KDTreeNode(m_dataset, this);
		KDTreeNode right = new KDTreeNode(m_dataset, this);
		while (it.hasNext()) {
			TreeItem inst = it.next();
			if (inst.instance().value(m_split_index) <= m_split_value)
				left.add(inst);
			else if (inst.instance().value(m_split_index) > m_split_value)
				right.add(inst);
		}
		m_left = left;
		if (left.size() == 0)
			throw new RuntimeException("Empty left leaf");
		m_right = right;
		if (right.size() == 0)
			throw new RuntimeException("Empty right leaf");
		m_values.clear();
		m_is_leaf = false;
	}


	


	public void print(PrintStream ps, int step) {
		for (int i = 0; i < step; i++)
			ps.print(" ");
		ps.print(m_is_leaf + " split dim " + m_split_index + " value "
				+ m_split_value + " size " + size());
		if (m_is_leaf) {
			ps.print(" nodes ");
			Iterator<TreeItem> it = m_values.iterator();
			while (it.hasNext())
				System.out.print(it.next().id() + " ");
			System.out.println();
		} else {
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

	private void clearRanges() {
		m_split_index = -1;
		m_split_value = Double.MAX_VALUE;
		m_max_distance = Double.MIN_VALUE;
		m_ranges = new Range[m_dataset.numAttributes()];
		for (int i = 0; i < m_ranges.length; i++)
			m_ranges[i] = new Range();
	}
	
	private void rescanRanges() {
		clearRanges();
		Iterator<TreeItem> it = m_values.iterator();
		while (it.hasNext())
			updateRanges(it.next());

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


	public void printRanges() {
		System.out.println("Ranges: index " + m_split_index + " value "
				+ m_split_value + " distance " + m_max_distance);
		int i = 0;
		for (Range r : m_ranges) {
			System.out.println((i++) + " " + r.m_min + " <x< " + r.m_max);
		}
	}

	public int getSplitIndex() {
		return m_split_index;
	}

	public double getSplitValue() {
		return m_split_value;
	}

	public KDTreeNode left() {
		return m_left;
	}

	public KDTreeNode right() {
		return m_right;
	}

	}