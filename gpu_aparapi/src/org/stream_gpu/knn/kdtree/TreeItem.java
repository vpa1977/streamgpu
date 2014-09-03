package org.stream_gpu.knn.kdtree;

import weka.core.Instance;

public class TreeItem {
	
	private long m_id;
	private GpuInstance m_instance;
	private KDTreeNode m_owner;
	
	public TreeItem( long id, GpuInstance inst)
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

	
	public Instance instance() 
	{
		return m_instance.wekaInstance();
	}
	
	public GpuInstance gpuInstance() {
		return m_instance;
	}

	public long id() {
		// TODO Auto-generated method stub
		return m_id;
	}
	
	public KDTreeNode owner()
	{
		return m_owner;
	}
	
	public void setOwner(KDTreeNode node) 
	{
		m_owner = node;
	}
}
