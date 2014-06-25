package org.stream_gpu.knn;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;

public class WithGui {

	/**
	 * @param args
	 */
	public static void main(final String[] args) {
		
		System.out.println(System.getProperty("java.class.path"));
		JFrame frame = new JFrame();
		
		JButton magic = new JButton("oh lol");
		magic.addActionListener(new ActionListener(){

			@Override
			public void actionPerformed(ActionEvent arg0) {
				try {
					DualTest.main(args);
				} catch (Throwable e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
			
		});
		frame.getContentPane().add(magic);
		frame.setVisible(true);

	}

}
