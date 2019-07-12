package model;

public class Feedback {
    private int user;
    private int item;
    private float rating;


    public Feedback(int user, int item, float rating) {
        this.user = user;
        this.item = item;
        this.rating = rating;
    }

    public int getUser() {
        return user;
    }

    public void setUser(int user) {
        this.user = user;
    }

    public int getItem() {
        return item;
    }

    public void setItem(int item) {
        this.item = item;
    }

    public float getRating() {
        return rating;
    }

    public void setRating(float rating) {
        this.rating = rating;
    }
}
