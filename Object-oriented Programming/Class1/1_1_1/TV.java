class TV {
    int size;
    String currentChannel;

    public TV(int size, String initialChannel) {
        this.size = size;
        this.currentChannel = initialChannel;
    }

    public void switchChannel(String newChannel) {
        this.currentChannel = newChannel;
        System.out.println("switch to " + newChannel);
    }
}
