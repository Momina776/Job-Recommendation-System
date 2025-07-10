@app.route('/apply/<job_id>', methods=['POST'])
@login_required
def apply_job(job_id):
    print(f"Received application for job_id: {job_id}")
    print(f"Current user: {current_user.id}, type: {current_user.user_type}")
    
    if current_user.user_type != 'job_seeker':
        print("Error: User is not a job seeker")
        return jsonify({
            'success': False, 
            'message': 'Only job seekers can apply for jobs',
            'redirectUrl': url_for('dashboard')
        })
    
    try:
        # Validate job_id format
        try:
            job_object_id = ObjectId(job_id)
            print(f"Valid ObjectId: {job_object_id}")
        except Exception as e:
            print(f"Invalid job_id format: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Invalid job ID format.',
                'redirectUrl': url_for('jobs')
            })

        # Check if job exists and is open
        job = jobs_collection.find_one({
            '_id': job_object_id,
            'status': 'open'
        })
        
        if not job:
            return jsonify({
                'success': False,
                'message': 'Job not found or no longer accepting applications.',
                'redirectUrl': url_for('jobs')
            })
        
        # Check if already applied
        existing_application = applications_collection.find_one({
            'job_id': job_object_id,
            'job_seeker_id': ObjectId(current_user.id)
        })
        
        if existing_application:
            return jsonify({
                'success': False,
                'message': 'You have already applied for this job.',
                'redirectUrl': url_for('job_details', job_id=job_id)
            })
        
        # Get form data
        cover_letter = request.form.get('cover_letter', '').strip()
        
        # Validate cover letter
        if len(cover_letter) < 100:
            return jsonify({
                'success': False,
                'message': 'Your cover letter must be at least 100 characters long.',
                'error': 'VALIDATION_ERROR'
            })
            
        if len(cover_letter) > 5000:
            return jsonify({
                'success': False,
                'message': 'Your cover letter cannot exceed 5000 characters.',
                'error': 'VALIDATION_ERROR'
            })
        
        # Create application
        application_data = {
            'job_id': job_object_id,
            'job_seeker_id': ObjectId(current_user.id),
            'employer_id': job['employer_id'],
            'date_applied': datetime.utcnow(),
            'status': 'pending',
            'cover_letter': cover_letter
        }
        
        try:
            # Insert application with retry logic
            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries):
                try:
                    result = applications_collection.insert_one(application_data)
                    if result.inserted_id:
                        print("Application inserted successfully with ID:", result.inserted_id)
                        return jsonify({
                            'success': True,
                            'message': 'Application submitted successfully!',
                            'redirectUrl': url_for('my_applications')
                        })
                    break
                except pymongo_errors.DuplicateKeyError:
                    print("Duplicate application attempt")
                    return jsonify({
                        'success': False,
                        'message': 'You have already applied for this job.',
                        'error': 'DUPLICATE_APPLICATION'
                    })
                except pymongo_errors.ConnectionFailure as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            # If we get here without returning, something went wrong
            raise Exception('Failed to insert application after multiple attempts')
            
        except pymongo_errors.ConnectionFailure:
            print("MongoDB connection error")
            return jsonify({
                'success': False,
                'message': 'Could not connect to the database. Please try again later.',
                'error': 'CONNECTION_ERROR'
            })
            
    except Exception as e:
        print(f"Error applying for job: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred while submitting your application. Please try again.',
            'error': 'SERVER_ERROR'
        })
