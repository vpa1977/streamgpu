package org.stream_gpu.knn.kdtree;

class Range {
	public double m_max;
	public double m_min;

	public Range() {
		m_min = Double.MAX_VALUE;
		m_max = -Double.MAX_VALUE;
	}
}