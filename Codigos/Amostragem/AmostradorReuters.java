import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;

public class AmostradorReuters {
	
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
		int qtdFiles = 6000;
		File folder;
		File[] listOfFiles;
		double porcentagem = 0.4;
		File novaFile;
		
		int quantFile = (int) (porcentagem * qtdFiles);
		
		folder = new File("C:/Users/Trevisan/Desktop/5 sem/Inteligência Artificial/trabalho/Trabalho-de-IA/reuters/text");
		listOfFiles = folder.listFiles();
		
		for (File file : listOfFiles) {
		    if (file.isFile()){
		    	if(quantFile == 0)break;
		    	
		    	novaFile = new File("C:/Users/Trevisan/Desktop/reuters amostra/" + file.getName());
		    	
		    	copyFileUsingChannel(file, novaFile);
		    	
		    	quantFile--;
		    }
		}
	}
}
