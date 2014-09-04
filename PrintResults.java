import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;


public class PrintResults {
	
	static final String HEADER_START = "Test :";
	static final String HEADER_END = "evaluation instances,total train time";
	static final String SECOND_HEADER = "---------------------------------------------------------------------------";
	
	enum Mode {
		idle, 
		header, 
		header2,
		numbers
	}
	
	public static void main(String[] args) throws Throwable
	{
		Mode mode = Mode.idle;
		boolean first_header = false;
		BufferedReader r = new BufferedReader(new FileReader(args[0]));
		ArrayList<Double> trainingTimes = new ArrayList<Double>();
		ArrayList<Double> testingTimes = new ArrayList<Double>();
		// Test : window=128 k =16
		// : test_size=10000 train_size =10000000
		//		 Stream: class moa.streams.generators.RandomRBFGenerator
		//		 ------Classifier:org.stream_gpu.knn.KnnGpuClassifier
		//		 evaluation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
		String s;
		int window = 0;
		while ( (s = r.readLine())  != null ) 
		{
			if (s.startsWith(HEADER_START))
			{
			//	System.out.println(SECOND_HEADER);
			//	System.out.println(SECOND_HEADER);
				mode = Mode.header;
				first_header = true;
				int eq = s.indexOf('=');
				int space = s.indexOf(' ', eq);
				window = Integer.parseInt(s.substring(eq+1, space));
				
			}
			if (s.startsWith(HEADER_END))
			{
				mode = Mode.numbers;
				continue;
			}
			if (s.startsWith(SECOND_HEADER))
			{
				if (first_header)
					mode = Mode.header2;
				else
					mode = Mode.idle;
				first_header = false;
				
				dumpNumbers(mode, window,testingTimes, trainingTimes);
				trainingTimes = new ArrayList<Double>();
				testingTimes = new ArrayList<Double>();
			}
			
			switch (mode)
			{
			case idle:
				break;
			case header:
			//	System.out.println(s);
				break;
			case header2:
			//	System.out.println(s);
				break;
			case numbers:
				collectNumbers(s,testingTimes, trainingTimes);
				break;
			}
			
		}
	}

	private static void dumpNumbers(Mode mode,int window, ArrayList<Double> testingTimes,
			ArrayList<Double> trainingTimes) {
		Collections.sort(testingTimes);
		Collections.sort(trainingTimes);
		testingTimes.remove(0);
		testingTimes.remove( testingTimes.size()-1);
		
		trainingTimes.remove(0);
		trainingTimes.remove(trainingTimes.size()-1);
		
		double trainMedian = 0;
		double testMedian = 0;
//		If n is odd then Median (M) = value of ((n + 1)/2)th item term.
//		If n is even then Median (M) = value of [((n)/2)th item term + ((n)/2 + 1)th item term ]/2
		testMedian = median(testingTimes);
		trainMedian = median(trainingTimes);
		if (mode.equals(Mode.idle))
		{
			System.out.println(testMedian + ","+ trainMedian);
		}
		else
			System.out.print(window + "," + testMedian + ","+ trainMedian + ",");

		
	}

	private static double median(ArrayList<Double> test) {
		double testMedian;
		if (test.size() % 2 == 0)
		{
			int index = test.size() / 2; 
			testMedian = test.get(index) + test.get(index + 1);
			testMedian = testMedian/2;
		}
		else
		{
			int index = (test.size() +1) / 2;
			testMedian = test.get(index);
		}
		return testMedian;
	}

	private static void collectNumbers(String s,ArrayList<Double> testingTimes,
			ArrayList<Double> trainingTimes) {
		StringTokenizer tk = new StringTokenizer( s, ",");
			//evaluation instances,
			tk.nextToken();
			//total train time,
			tk.nextToken();
			//total train speed
			tk.nextToken();
			//last train time
			tk.nextToken();
			//last train speed,
			trainingTimes.add(Double.parseDouble(tk.nextToken()));
			//test time
			tk.nextToken();
			//test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
			testingTimes.add(Double.parseDouble(tk.nextToken()));
			
			
	}

}
