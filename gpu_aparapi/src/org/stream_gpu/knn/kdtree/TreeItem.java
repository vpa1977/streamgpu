package org.stream_gpu.knn.kdtree;

import weka.core.Instance;

public class TreeItem {
	public TreeItem( long id, Instance inst)
	{
		m_id = id;
		m_instance = inst;
	}
	
	@Override
	public boolean equals(Object other) {
		if (other instanceof TreeItem)
			return ((TreeItem)other).m_id == m_id;
		return false;
	}
	@Override
	public int hashCode() {
		return (int)m_id;
	}

	private long m_id;
	private Instance m_instance;
	public Instance instance() {
		return m_instance;
	}

	public long id() {
		// TODO Auto-generated method stub
		return m_id;
	}
}
