import java.io.*;
import java.nio.channels.FileChannel;

public class AmostradorBBC{
	
	private static void copyFileUsingChannel(File source, File dest) throws IOException{
	    FileChannel sourceChannel = null;
	    FileChannel destChannel = null;
	    try {
	        sourceChannel = new FileInputStream(source).getChannel();
	        destChannel = new FileOutputStream(dest).getChannel();
	        destChannel.transferFrom(sourceChannel, 0, sourceChannel.size());
	       }finally{
	           sourceChannel.close();
	           destChannel.close();
	       }
	}
	
	public static void main(String[] args) throws IOException{
		String[] diretorios = {"business", "entertainment", "politics", "sport", "tech"};
		int[] qtdFiles = {510, 386, 417, 511, 401};
		File folder;
		File[] listOfFiles;
		double porcentagem = 0.4;
		int quantFile;
		File novaFile;
		
		
		for(int i = 0; i < diretorios.length; i++){
			folder = new File("C:/Users/Trevisan/Desktop/5 sem/Inteligência Artificial/trabalho/Trabalho-de-IA/bbc/" + diretorios[i]);
			listOfFiles = folder.listFiles();
			
			quantFile = (int) (porcentagem * qtdFiles[i]);
			
			for (File file : listOfFiles) {
			    if (file.isFile()){
			    	if(quantFile == 0)break;
			    	
			    	novaFile = new File("C:/Users/Trevisan/Desktop/Amostra BBC/" + diretorios[i] + " " + file.getName());
			    	
			    	copyFileUsingChannel(file, novaFile);
			    	
			    	quantFile--;
			    }
			}
		}
	}
}