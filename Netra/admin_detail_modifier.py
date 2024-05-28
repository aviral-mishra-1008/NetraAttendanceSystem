import pickle 

l = [('20225017',False),('20225022',False)]

with open('Admin Details\\dummy-student_details.pkl','wb') as f:
    pickle.dump(l,f)

