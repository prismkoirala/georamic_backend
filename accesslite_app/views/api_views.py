from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
# from .supabase_client import SupabaseClient
from accesslite_app.services import main_calc

class CalcISOAPIView(APIView):
    
    features_to_fetch=["water","school","park"]
    time_budget = 30


    def get(self, request):
        """
        Handle GET requests to provide instructions or an error.
        """
        return Response(
            {
                'message': 'This endpoint requires a POST request with parameters: lat, lng, mode.',
                'example': {
                    'lat': 40.7128,
                    'lng': -74.0060,
                    'mode': 'driving'
                }
            },
            status=status.HTTP_200_OK
        )

    def post(self, request):

        """
        Handle POST requests from frontend
        """
                
        lat = request.data.get('lat')
        lng = request.data.get('lng')
        mode = request.data.get('mode')

        if not all([lat, lng, mode]):
            return Response(
                {'error': 'Missing required parameters: lat, lng, mode'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            lat = float(lat)
            lng = float(lng)
        except (ValueError, TypeError):
            return Response(
                {'error': 'Invalid lat or lng: must be numeric'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        res = main_calc(lat, lng, self.time_budget, mode, self.features_to_fetch)
        
        iso_result = {
            'lat': lat,
            'lng': lng,
            'mode': mode,
            'result': res
        }

        return Response(iso_result, status=status.HTTP_200_OK)