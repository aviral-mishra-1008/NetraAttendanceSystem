import pickle 

students = [('20225018',False),('20223301',False),('20223298',False),('20223302',False)]
professors = [('20225017',False)]
with open('Admin Details\\dummy-student_details.pkl','wb') as f:
    pickle.dump(students,f)

with open('Admin Details\\dummy-prof_details.pkl','wb') as f:
    pickle.dump(professors,f)

