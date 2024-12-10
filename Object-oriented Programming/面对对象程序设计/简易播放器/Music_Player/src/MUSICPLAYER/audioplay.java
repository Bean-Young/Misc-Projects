package MUSICPLAYER;
import javax.sound.sampled.*;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

class audioplay {
    AudioInputStream audioInputStream;
    Clip clip;
    URL url;
    boolean adcFlag = false;
    boolean playFlag = false;
    public void SetPlayAudioPath(String path) {
        try {
            url = new URL(path);
            if (adcFlag == true) {
                clip.stop();
                playFlag = false;
            }
            audioInputStream = AudioSystem.getAudioInputStream(url);
            clip = AudioSystem.getClip();
            clip.open(audioInputStream);
            adcFlag = true;
        } catch (MalformedURLException e1) {
            e1.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (LineUnavailableException e) {
            e.printStackTrace();
        } catch (UnsupportedAudioFileException e) {
			e.printStackTrace();
		}
    }


    public void play() {
        if (clip != null && adcFlag == true) {
            clip.start();
            playFlag = true;
        }
    }

    public void stop() {
        if (clip != null) {
            clip.stop();
            playFlag = false;
        }
    }
    public static void main(String[] args) {
        audioplay audioPlayer = new audioplay();
        audioPlayer.SetPlayAudioPath("E://Java//Music_Player//resource//飘洋过海.wav");
        audioPlayer.SetPlayAudioPath("E://Java//Music_Player//resource//山外小楼夜听雨.wav");
        audioPlayer.SetPlayAudioPath("E://Java//Music_Player//resource//我和我的祖国.wav");
        audioPlayer.play();
   }
}