import pickle

if __name__ == '__main__':
    mnb = pickle.load(open("model.pkl", "rb"))

    print(mnb.predict("Congratulations love!"))