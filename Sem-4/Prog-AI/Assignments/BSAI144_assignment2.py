"""Q1. Develop a Python program to manage student records using a dictionary of tuples. Each student 
 is uniquely identified by a student ID, which serves as the key in the dictionary. The values 
 associated with each student ID are tuples containing information about the student, including 
 their name, age, and course.

 a. Describe how you would implement functions for inserting, updating, and deleting 
 student records when the user provides the data. The insert_student_record function 
 should allow the user to add new records, the update_student_record function should 
 enable the user to modify existing records, and the delete_student_record function 
 should allow the user to remove records.

 b. Provide an example code snippet for these functions and demonstrate their usage with 
 user-provided student records. Include sample operations, such as inserting a new 
 student record, updating an existing record, and deleting a record, where the user is 
 prompted to provide the necessary information.

 c. After performing these operations with user-provided data, print the updated student 
 records to show the changes made using these functions.

 d. Please outline the steps and code for this student records management system, taking 
 into account that the user will provide the record data."""
 
def insert_student_record(records,student_id,student_name,age,course):
    records[student_id]=(student_name,age,course)

def update_student_record(records,student_id,student_name,age,course):
    if student_id in records:
       records[student_id]=(student_name,age,course)

def del_student_record(records,student_id):
    if student_id in records:
        del records[student_id]
    else:
        print("student id not found")

def display(records):
    for student_id,record in records.items():
        print(f'''student id:{student_id}\tstudent name:{record[0]}\tstudent age:{record[1]}\tcourse:{record[2]}''')
def main():
  student_record={}
  while True:
    print("***********  Student record Management system  ****************")
    print("Menu")
    print("\tpress 1 for insert record")
    print("\tpress 2 for update record")
    print("\tpress 3 for delete record")
    print("\tpress 4 for display record")
    choice=input("enter your choice:")
    if choice=='1':
           student_id=int(input("enter student id:"))
           student_name=input("enter student_name:")
           age=int(input("enter student age:"))
           course=input("enter student course:")
           insert_student_record(student_record,student_id,student_name,age,course)
           
    elif choice=='2':
           student_id=input("enter student id:")
           student_name=input("enter student_name:")
           age=int(input("enter student age:"))
           course=input("enter student course:")
           update_student_record(student_record,student_id,student_name,age,course)
           
    elif choice=='3':
           student_id=int(input("enter student id to delete record:"))
           del_student_record(student_record,student_id)
        
    elif choice=='4':
           print(student_record)

    else:
           print("SORRY !! invalid choice") 
 
 

""" Q2. Develop a Python program for a music library. You want to manage the playlists and songs in the 
 library efficiently. Each song in the library is represented as a dictionary containing the song'

 title, artist, and duration. Playlists are represented as lists of song dictionaries. You need to 
 implement various operations to manage playlists and songs within the library.
 
 a. Define a Python function to create a new playlist. The function should take the name of 
 the playlist as input and return an empty playlist (an empty list of songs).
 6
 b. Implement a function to add a song to a playlist. This function should take the playlist, 
 song title, artist, and duration as input and add a new song dictionary to the playlist.

 c. Create a function to remove a song from a playlist. This function should take the playlist 
 and the title of the song to be removed. Ensure that the song is removed from the 
 playlist, and if the song is not found, provide a suitable message.

 d. Develop a function to search for songs by a specific artist. This function should take the 
 playlist and the artist's name as input, returning a new list containing all songs by that 
 artist.

 e. Implement a function to calculate the total duration of a playlist. This function should 
 take the playlist as input and return the total duration of all songs in the playlist.

 f. Finally, create a program that allows users to interact with the music library. Users can 
create playlists, add songs to playlists, remove songs, search for songs by artist, and 
calculate the total duration of a playlist. Demonstrate the use of these functions with 
sample operations and print the resulting playlists and song details."""

def create_playlist(name):
    return []

def add_song_to_playlist(playlist, title, artist, duration):
    song = {'title': title, 'artist': artist, 'duration': duration}
    playlist.append(song)

def remove_song_from_playlist(playlist, title):
    for song in playlist:
        if song['title'] == title:
            playlist.remove(song)
            print(f"Song '{title}' removed from the playlist.")
            return
    print(f"Song '{title}' not found in the playlist.")

def search_songs_by_artist(playlist, artist):
    songs_by_artist = []
    for song in playlist:
        if song['artist'] == artist:
            songs_by_artist.append(song)
    return songs_by_artist

def calculate_playlist_duration(playlist):
    total_duration = 0
    for song in playlist:
        total_duration += song['duration']
    return total_duration

def main():
    library = {}

    while True:
        print("\nMusic Library Menu")
        print("1. Create Playlist")
        print("2. Add Song to Playlist")
        print("3. Remove Song from Playlist")
        print("4. Search Songs by Artist")
        print("5. Calculate Playlist Duration")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            playlist_name = input("Enter the name of the playlist: ")
            library[playlist_name] = create_playlist(playlist_name)
            print(f"Playlist '{playlist_name}' created successfully.")
        elif choice == '2':
            playlist_name = input("Enter the name of the playlist: ")
            if playlist_name not in library:
                print("Playlist not found.")
                continue
            title = input("Enter song title: ")
            artist = input("Enter artist name: ")
            duration = int(input("Enter song duration (in seconds): "))
            add_song_to_playlist(library[playlist_name], title, artist, duration)
            print(f"Song '{title}' added to the playlist '{playlist_name}'.")
        elif choice == '3':
            playlist_name = input("Enter the name of the playlist: ")
            if playlist_name not in library:
                print("Playlist not found.")
                continue
            title = input("Enter the title of the song to remove: ")
            remove_song_from_playlist(library[playlist_name], title)
        elif choice == '4':
            playlist_name = input("Enter the name of the playlist: ")
            if playlist_name not in library:
                print("Playlist not found.")
                continue
            artist = input("Enter the name of the artist: ")
            songs_by_artist = search_songs_by_artist(library[playlist_name], artist)
            if songs_by_artist:
                print(f"Songs by artist '{artist}':")
                for song in songs_by_artist:
                    print(f"Title: {song['title']}, Artist: {song['artist']}, Duration: {song['duration']} seconds")
            else:
                print(f"No songs found by artist '{artist}' in the playlist '{playlist_name}'.")
        elif choice == '5':
            playlist_name = input("Enter the name of the playlist: ")
            if playlist_name not in library:
                print("Playlist not found.")
                continue
            total_duration = calculate_playlist_duration(library[playlist_name])
            print(f"Total duration of the playlist '{playlist_name}': {total_duration} seconds")
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

