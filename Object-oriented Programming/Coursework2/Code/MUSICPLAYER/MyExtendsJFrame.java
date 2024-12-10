package MUSICPLAYER;

import javax.swing.*;
import javax.swing.text.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

@SuppressWarnings({"serial", "rawtypes", "unchecked"})
public class MyExtendsJFrame extends JFrame implements ActionListener, MouseListener {
    final int width = 1280;
    final int height = 836;
    final int tool_width = 42;

    audioplay audioPlay;
    Timer nTimer;
    String MusicName;
    String playFile;
    String playFileName;
    String playFileDirectory = "E://Java//Music_Player//resource//";
    int MusicTime;
    Vector vt = new Vector();
    Vector vtime = new Vector();
    int flagway = 0;

    JLabel background;
    JButton buttonPlay;
    JButton buttonOpenFile;
    JTextPane textLyrics;
    JLabel playTime;
    JList listPlayFile;
    JButton buttonNext;
    JButton buttonPre;
    JLabel backgroundPlay;
    JTextArea textMusic;
    JList listPlayFileTime;
    JButton buttonShowList;
    JTextArea musictitle;
    JButton buttonWay;
    JTextArea TimeCount;
    JLabel gifwave;

    String[] sLyrics1 = {"记忆它总是慢慢的累积\n", "在我心中无法抹去\n", "为了你的承诺\n", "我在最绝望的时候\n", "都忍着不哭泣\n", "陌生的城市啊\n", "熟悉的角落里\n", "也曾彼此安慰\n", "也曾相拥叹息\n", "不管将会面对什么样的结局\n", "在漫天风沙里望着你远去\n", "我竟悲伤得不能自己\n", "多盼能送君千里\n", "直到山穷水尽\n", "一生和你相依"};

    String[] sLyrics2 = {"芙蓉花又栖满了枝头 \n", "亲何蝶雅留\n", "票白如江水向东流入\n", "望断门前隔岸的杨柳 \n", "寂寞仍不休\n", "我无言让眼泪长流\n", "我独酌山外小阁楼\n", "听一夜相思愁\n", "醉后让人烦忧心事雅收\n", "山外小阁楼我乘一叶小舟\n", "放思念随风漂流\n", "我独坐山外小阁楼\n", "窗外渔火如豆\n", "江畔晚风拂柳诉尽离愁\n", "当月色暖小楼是谁又在弹奏\n", "那一曲思念常留\n"};

    String[] sLyrics3 = {"我和我的祖国\n", "一刻也不能分割\n", "无论我走到哪里\n", "都流出一首赞歌\n", "我歌唱每一座高山\n", "我歌唱每一条河\n", "袅袅炊烟，小小村落\n", "路上一道辙\n", "你用你那母亲的脉搏和我诉说\n"};

    public MyExtendsJFrame() {
        audioPlay = new audioplay();
        setTitle("播放器");
        setBounds(160, 100, 1300, 880);
        setLayout(null);
        init();
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    void init() {
        ImageIcon img = new ImageIcon(playFileDirectory + "background.jpg");
        JLabel background = new JLabel(img);
        background.setBounds(0, -100, width, height);
        getLayeredPane().add(background, new Integer(Integer.MIN_VALUE));
        ((JComponent) getContentPane()).setOpaque(false);

        buttonPlay = new JButton();
        buttonPlay.setBounds(width / 2 - tool_width / 2, height - tool_width - 335, tool_width, tool_width);
        Icon icon = new ImageIcon(playFileDirectory + "play.jpg");
        buttonPlay.setIcon(icon);
        buttonPlay.setBorderPainted(false);
        buttonPlay.addActionListener(this);
        add(buttonPlay);

        buttonNext = new JButton("");
        buttonNext.setBounds(width / 2 - tool_width / 2 + 40, height - tool_width - 335, tool_width, tool_width);
        icon = new ImageIcon(playFileDirectory + "next.jpg");
        buttonNext.setIcon(icon);
        buttonNext.setBorderPainted(false);
        buttonNext.addActionListener(this);
        add(buttonNext);

        buttonPre = new JButton("");
        buttonPre.setBounds(width / 2 - tool_width / 2 - 40, height - tool_width - 335, tool_width, tool_width);
        icon = new ImageIcon(playFileDirectory + "prev.jpg");
        buttonPre.setIcon(icon);
        buttonPre.setBorderPainted(false);
        buttonPre.addActionListener(this);
        add(buttonPre);

        buttonWay = new JButton("");
        buttonWay.setBounds(width / 2 - tool_width / 2 - 80, height - tool_width - 335, tool_width, tool_width);
        icon = new ImageIcon(playFileDirectory + "loop.jpg");
        buttonWay.setIcon(icon);
        buttonWay.setBorderPainted(false);
        buttonWay.addActionListener(this);
        add(buttonWay);

        icon = new ImageIcon(playFileDirectory + "background.jpg");
        backgroundPlay = new JLabel(icon);
        backgroundPlay.setBounds(518, 191, 240, 236);
        getLayeredPane().add(backgroundPlay);

        gifwave = new JLabel();
        gifwave.setBounds(518, 191, 240, 236);
        add(gifwave);


        buttonOpenFile = new JButton("");
        buttonOpenFile.setBounds(width / 2 - tool_width / 2 + 80, height - tool_width - 335, tool_width, tool_width);
        icon = new ImageIcon(playFileDirectory + "open.jpg");
        buttonOpenFile.setIcon(icon);
        buttonOpenFile.setBorderPainted(false);
        buttonOpenFile.addActionListener(this);
        add(buttonOpenFile);

        icon = new ImageIcon(playFileDirectory + "time.jpg");
        playTime = new JLabel(icon);
        playTime.setBounds(0, height - 394, 0, 3);
        add(playTime);

        TimeCount = new JTextArea("00:00");
        TimeCount.setBounds(width / 2 - tool_width / 2 - 180, height - tool_width - 325, 80, 20);
        TimeCount.setForeground(Color.white);
        TimeCount.setFont(new Font("楷体", 1, 20));
        TimeCount.setOpaque(false);
        add(TimeCount);

        buttonShowList = new JButton("");
        buttonShowList.setBounds(width / 2 - tool_width / 2 + 400, height - tool_width - 325, tool_width - 3, tool_width - 3);
        buttonShowList.setIcon(new ImageIcon(playFileDirectory + "list.jpg"));
        buttonShowList.setBorderPainted(false);
        buttonShowList.addActionListener(this);
        add(buttonShowList);

        textLyrics = new JTextPane();
        textLyrics.setBounds(width / 2 - 400, height / 2 - 225, 250, 300);
        textLyrics.setForeground(Color.white);
        textLyrics.setOpaque(false);
        add(textLyrics);
        textLyrics.setText("");
        textLyrics.setFont(new Font("楷体", 1, 20));

        musictitle = new JTextArea("");
        musictitle.setBounds(width / 2 - 400, height / 2 - 290, 300, 100);
        musictitle.setForeground(Color.white);
        musictitle.setOpaque(false);
        musictitle.setFont(new Font("楷体", 1, 30));
        add(musictitle);

        listPlayFile = new JList();
        listPlayFile.setBounds(800, height - 610, 200, 150);
        listPlayFile.setOpaque(false);
        listPlayFile.setFont(new Font("楷体", 1, 20));
        listPlayFile.setBackground(new Color(0, 0, 0, 0));
        listPlayFile.setForeground(Color.white);
        add(listPlayFile);
        listPlayFile.addMouseListener(this);

        listPlayFileTime = new JList();
        listPlayFileTime.setBounds(width - 280, height - 610, 150, 150);
        listPlayFileTime.setOpaque(false);
        listPlayFileTime.setFont(new Font("楷体", 1, 20));
        listPlayFileTime.setBackground(new Color(0, 0, 0, 0));
        listPlayFileTime.setForeground(Color.white);
        add(listPlayFileTime);
    }
    void updateGif() {
        if (playFileName.equals("我和我的祖国.wav")) {
            gifwave.setIcon(new ImageIcon(playFileDirectory + "playgif5.gif"));
        } else if (playFileName.equals("山外小楼夜听雨.wav")) {
            gifwave.setIcon(new ImageIcon(playFileDirectory + "playgif6.gif"));
        } else if (playFileName.equals("飘洋过海.wav")) {
            gifwave.setIcon(new ImageIcon(playFileDirectory + "playgif7.gif"));
        }
        gifwave.setVisible(true);
    }

    public void timerFun(int musicTime) {
        MusicTime = musicTime;
        if (nTimer != null) {
            nTimer.cancel();
        }

        nTimer = new Timer();
        nTimer.schedule(new TimerTask() {
            int PlayTime = 0;

            public void run() {
                PlayTime++;
                if (PlayTime >= MusicTime) {
                    nTimer.cancel();
                    gifwave.setVisible(false); // Stop the animation when the song ends

                    if (flagway == 0 && vt.size() != 0) {
                        audioPlay.play();
                        timerFun(MusicTime);
                    } else if (flagway == 1 && vt.size() != 0) {
                        int position = vt.lastIndexOf(playFileName);
                        position = (position + 1) % (vt.size());
                        playFileName = (String) vt.get(position);
                        playFile = playFileDirectory + playFileName;
                        audioPlay.SetPlayAudioPath("file:" + playFile);
                        audioPlay.play();
                        File file = new File(playFile);
                        int nMusicTime = (int) file.length() / 1024 / 173;
                        timerFun(nMusicTime);
                    } else if (flagway == 2) {
                        int position = vt.lastIndexOf(playFileName);
                        int choose = 0;
                        do {
                            choose = (int) (Math.random() * vt.size());
                        } while (position == choose);
                        playFileName = (String) vt.get(choose);
                        playFile = playFileDirectory + playFileName;
                        audioPlay.SetPlayAudioPath("file:" + playFile);
                        audioPlay.play();
                        File file = new File(playFile);
                        int nMusicTime = (int) file.length() / 1024 / 173;
                        timerFun(nMusicTime);
                    }
                }

                int Second = PlayTime % 60;
                int Minute = PlayTime / 60;

                String sSecond = "";
                String sMinute = "";
                if (Second < 10) {
                    sSecond = "0" + Second;
                } else {
                    sSecond = "" + Second;
                }
                if (Minute < 10) {
                    sMinute = "0" + Minute;
                } else {
                    sMinute = "" + Minute;
                }
                String sPlayTime = sMinute + ":" + sSecond;
                TimeCount.setText(sPlayTime);
                playTime.setBounds(205, height - 394, width * PlayTime / MusicTime, 3);
                textLyrics.setText("");
                int flag = 0;
                if (playFileName.equals("飘洋过海.wav")) {
                    flag = 1;
                } else if (playFileName.equals("山外小楼夜听雨.wav")) {
                    flag = 2;
                } else if (playFileName.equals("我和我的祖国.wav")) {
                    flag = 3;
                }

                if (flag == 1) {
                    int[] breaktime = {1, 5, 8, 11, 13, 17, 23, 27, 30, 32, 36, 40, 43, 45, 49, 53};
                    final int MAX = 12;
                    int position = 0;
                    for (int i = 0; i < sLyrics1.length; i++) {
                        for (position = 0; position < sLyrics1.length - 1; position++) {
                            if (PlayTime < breaktime[0]) {
                                break;
                            }
                            if (PlayTime >= breaktime[position] && PlayTime <= breaktime[position + 1] - 1) {
                                break;
                            }
                        }
                        SimpleAttributeSet attrSet = new SimpleAttributeSet();
                        StyleConstants.setFontFamily(attrSet, "隶书");
                        StyleConstants.setFontSize(attrSet, 13);
                        int over = position - MAX;
                        try {
                            Document doc = MyExtendsJFrame.this.textLyrics.getDocument();
                            StyleConstants.setForeground(attrSet, Color.yellow);
                            StyleConstants.setBold(attrSet, true);
                            if (PlayTime >= breaktime[i] && PlayTime <= breaktime[i + 1] - 1) {
                                doc.insertString(doc.getLength(), sLyrics1[i], attrSet);
                            } else {
                                StyleConstants.setForeground(attrSet, Color.white);
                                StyleConstants.setBold(attrSet, false);
                                if (over >= 0 && i - over <= 0) {
                                    continue;
                                }
                                doc.insertString(doc.getLength(), sLyrics1[i], attrSet);
                            }
                        } catch (BadLocationException localBadLocationException) {
                        }
                    }
                } else if (flag == 2) {
                    int[] breaktime = {17, 21, 25, 30, 35, 39, 44, 49, 53, 59, 66, 72, 76, 81, 87, 94, 100};
                    final int MAX = 12;
                    int position = 0;

                    for (int i = 0; i < sLyrics2.length; i++) {
                        for (position = 0; position < sLyrics2.length - 1; position++) {
                            if (PlayTime < breaktime[0]) {
                                break;
                            }
                            if (PlayTime >= breaktime[position] && PlayTime <= breaktime[position + 1] - 1) {
                                break;
                            }
                        }
                        SimpleAttributeSet attrSet = new SimpleAttributeSet();
                        StyleConstants.setFontFamily(attrSet, "隶书");
                        StyleConstants.setFontSize(attrSet, 12);
                        int over = position - MAX;
                        try {
                            Document doc = MyExtendsJFrame.this.textLyrics.getDocument();
                            StyleConstants.setForeground(attrSet, Color.yellow);
                            StyleConstants.setBold(attrSet, true);
                            if (PlayTime >= breaktime[i] && PlayTime <= breaktime[i + 1] - 1) {
                                doc.insertString(doc.getLength(), sLyrics2[i], attrSet);
                            } else {
                                StyleConstants.setForeground(attrSet, Color.white);
                                StyleConstants.setBold(attrSet, false);
                                if (over >= 0 && i - over <= 0) {
                                    continue;
                                }
                                doc.insertString(doc.getLength(), sLyrics2[i], attrSet);
                            }
                        } catch (BadLocationException localBadLocationException) {
                        }
                    }
                } else if (flag == 3) {
                    int[] breaktime = {1, 6, 10, 15, 19, 24, 28, 32, 50, 59};
                    final int MAX = 12;
                    int position = 0;
                    for (int i = 0; i < sLyrics3.length; i++) {
                        for (position = 0; position < sLyrics3.length - 1; position++) {
                            if (PlayTime < breaktime[0]) {
                                break;
                            }
                            if (PlayTime >= breaktime[position] && PlayTime <= breaktime[position + 1] - 1) {
                                break;
                            }
                        }
                        SimpleAttributeSet attrSet = new SimpleAttributeSet();
                        StyleConstants.setFontFamily(attrSet, "隶书");
                        StyleConstants.setFontSize(attrSet, 20);
                        int over = position - MAX;
                        try {
                            Document doc = MyExtendsJFrame.this.textLyrics.getDocument();
                            StyleConstants.setForeground(attrSet, Color.yellow);
                            StyleConstants.setBold(attrSet, true);
                            if (PlayTime >= breaktime[i] && PlayTime <= breaktime[i + 1] - 1) {
                                doc.insertString(doc.getLength(), sLyrics3[i], attrSet);
                            } else {
                                StyleConstants.setForeground(attrSet, Color.white);
                                StyleConstants.setBold(attrSet, false);
                                if (over >= 0 && i - over <= 0) {
                                    continue;
                                }
                                doc.insertString(doc.getLength(), sLyrics3[i], attrSet);
                            }
                        } catch (BadLocationException localBadLocationException) {
                        }
                    }
                }
            }
        }, 0L, 1000L);
    }

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == buttonOpenFile) {
            FileDialog openFile = new FileDialog(this, playFileDirectory);
            openFile.setVisible(true);
            if (openFile.getFile() != null) {
                playFileName = openFile.getFile();
            } else {
                return;
            }
            playFile = playFileDirectory + playFileName;

            audioPlay.SetPlayAudioPath("file:" + playFile);
            audioPlay.stop();

            File file = new File(playFile);
            int iMusicTime = (int) file.length() / 1024 / 173;
            int iSecond = iMusicTime % 60;
            int iMinute = iMusicTime / 60;

            if (!vt.contains(playFileName)) {
                vt.add(playFileName);
                listPlayFile.setListData(vt);
                vtime.add(iMinute + ":" + iSecond);
                listPlayFileTime.setListData(vtime);
            }

            audioPlay.SetPlayAudioPath("file:" + this.playFile);
            audioPlay.play();
            musictitle.setText(this.playFileName);
            Icon icon = new ImageIcon(playFileDirectory + "stop.jpg");
            buttonPlay.setIcon(icon);
            backgroundPlay.setVisible(false);
            int nMusicTime = (int) file.length() / 1024 / 173;
            timerFun(nMusicTime);
        }

        if (e.getSource() == buttonPlay) {
            if (!audioPlay.playFlag) {
                if (vt.size() != 0) {
                    if (listPlayFile.getSelectedValue() != null) {
                        playFile = playFileDirectory + listPlayFile.getSelectedValue().toString();
                    } else {
                        playFile = playFileDirectory + listPlayFile.getModel().getElementAt(0).toString();
                        listPlayFile.setSelectedIndex(0);
                    }
                    audioPlay.stop();
                    audioPlay.SetPlayAudioPath("file:" + playFile);
                    audioPlay.play();
                    Icon icon = new ImageIcon(playFileDirectory + "stop.jpg");
                    buttonPlay.setIcon(icon);
                    backgroundPlay.setVisible(false);
                    File file = new File(this.playFile);
                    int nMusicTime = (int) file.length() / 1024 / 173;
                    timerFun(nMusicTime);
                } else {
                    System.out.println("没有音乐可以播放");
                }
            } else {
                audioPlay.stop();
                nTimer.cancel();
                Icon icon = new ImageIcon(playFileDirectory + "play.jpg");
                this.buttonPlay.setIcon(icon);
                this.backgroundPlay.setVisible(true);
            }
        }

        if (e.getSource() == buttonShowList) {
            if (listPlayFile.isVisible()) {
                listPlayFile.setVisible(false);
                listPlayFileTime.setVisible(false);
            } else {
                listPlayFile.setVisible(true);
                listPlayFileTime.setVisible(true);
            }
        }

        if (e.getSource() == buttonNext) {
            if (vt.size() != 0) {
                int position = vt.lastIndexOf(playFileName);
                position = (position + 1) % (vt.size());
                playFileName = (String) vt.get(position);
                playFile = playFileDirectory + playFileName;
                musictitle.setText(playFileName);
                audioPlay.SetPlayAudioPath("file:" + playFile);
                audioPlay.play();
                File file = new File(playFile);
                int nMusicTime = (int) file.length() / 1024 / 173;
                timerFun(nMusicTime);
            } else {
                System.out.println("没有音乐可以播放");
            }
        }

        if (e.getSource() == buttonPre) {
            if (vt.size() != 0) {
                int position = vt.lastIndexOf(playFileName);
                position = (vt.size() + position - 1) % (vt.size());
                playFileName = (String) vt.get(position);
                playFile = playFileDirectory + playFileName;
                musictitle.setText(playFileName);
                audioPlay.SetPlayAudioPath("file:" + playFile);
                audioPlay.play();
                File file = new File(playFile);
                int nMusicTime = (int) file.length() / 1024 / 173;
                timerFun(nMusicTime);
            } else {
                System.out.println("没有音乐可以播放");
            }
        }

        if (e.getSource() == buttonWay) {
            if (flagway == 0) {
                flagway = 1;
                Icon icon = new ImageIcon(playFileDirectory + "unloop.jpg");
                buttonWay.setIcon(icon);
            } else if (flagway == 1) {
                flagway = 0;
                Icon icon = new ImageIcon(playFileDirectory + "loop.jpg");
                buttonWay.setIcon(icon);
            }
        }
    }

    public void mouseClicked(MouseEvent e) {
        if (e.getClickCount() == 2) {
            if (e.getSource() == listPlayFile) {
                int n = vt.size();
                if (n != 0) {
                    if (listPlayFile.getSelectedValue() != null) {
                        playFileName = listPlayFile.getSelectedValue().toString();
                        playFile = playFileDirectory + playFileName;
                    } else {
                        playFileName = listPlayFile.getModel().getElementAt(0).toString();
                        playFile = playFileDirectory + playFileName;
                    }

                    audioPlay.SetPlayAudioPath("file:" + playFile);
                    audioPlay.play();

                    musictitle.setText(listPlayFile.getSelectedValue().toString());
                    Icon icon = new ImageIcon(playFileDirectory + "stop.jpg");
                    buttonPlay.setIcon(icon);
                    updateGif(); // Update the GIF animation based on the song

                    File file = new File(playFile);
                    int nMusicTime = (int) file.length() / 1024 / 173;
                    timerFun(nMusicTime);
                }
            }
        }
    }

    public void mousePressed(MouseEvent e) {
    }

    public void mouseReleased(MouseEvent e) {
    }

    public void mouseEntered(MouseEvent e) {
    }

    public void mouseExited(MouseEvent e) {
    }
}
