<template>
    <v-container>
        <v-layout row>
            <v-flex>
                <DocumentList />
            </v-flex>
        </v-layout>
    </v-container>
</template>

<script>
import DocumentList from '../components/DocumentList.vue';

export default {
    components: {
        DocumentList,
    },
};
</script>
