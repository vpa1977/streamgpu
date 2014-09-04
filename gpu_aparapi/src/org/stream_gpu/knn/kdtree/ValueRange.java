package org.stream_gpu.knn.kdtree;

class ValueRange {
	public double m_max;
	public double m_min;

	public ValueRange() {
		m_min = Double.MAX_VALUE;
		m_max = -Double.MAX_VALUE;
	}
}